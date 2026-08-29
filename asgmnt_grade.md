# Grading — Section 14 (Student Exercises)

**Submission:** `ppo_glassbox_llm-final_ilay_nisim_boris_kogan.ipynb`
**Students:** Ilay, Nisim, Boris, Kogan
**Assignment:** Section 14 — *Student exercises (predict → turn the dial → verify)*, exercises 1–5
**Graded:** 2026-07-04

## Overall grade: **A (95 / 100)**

An exemplary submission. All five exercises are completed with the exact
methodology the section asks for — a **prediction written before the run**, a clean
experiment that resets to `policy_fresh_state` for comparability, and a **verification
that honestly reconciles the result with the prediction** instead of pretending the
data matched. The intellectual honesty about single-seed noise and about *which* half
of each prediction actually surfaced is the standout quality — this is how the exercise
is meant to be done. The only thing separating this from a perfect score is that the
group repeatedly *identifies* the cleaner follow-up experiment and then stops at
describing it rather than running it.

---

## Per-exercise breakdown

### Exercise 1 — Clip ε · **19/20**
- Prediction is correct and complete: ε=0.05 → clip bites often → high `clipfrac`,
  cautious/slow; ε=0.6 → clip rarely bites → big risky steps; **ε=0.6 = faster-but-riskier**.
- Verification is genuinely sophisticated. Rather than hand-wave, they caught that the
  **ε=0.05-vs-0.2 `clipfrac` ordering is run-dependent** and gave the correct mechanism
  (a tighter ε clips at a lower threshold *but also takes smaller steps*, so the ratio
  drifts less — the two effects partly cancel). They pivoted to **entropy collapse** as
  the robust signal of "risk" (≈0.25 → 0.17 → 0.05 as ε grows). That is exactly right.
- Honest note that the "faster" half of the prediction never surfaced at a 60-iter
  reward ceiling.
- **Deduction:** they diagnose that a shorter horizon / harder reward would expose the
  speed gap, but don't actually run it. Naming the experiment ≠ running it.

### Exercise 2 — GAE λ · **19/20**
- Prediction correct: λ=0 high-bias/low-variance/smooth-but-can-stall; λ=1 unbiased/
  high-variance/noisier.
- Verification confirms the direction (λ=0 smoothest at step-noise ≈0.16; λ≥0.95 much
  noisier ≈0.6–0.8) and adds two honest nuances: (1) λ=0 is *so* biased it barely learns
  (stuck near the reward floor), and (2) λ=1.0 is **not** reliably noisier than λ=0.95 —
  that fine gradation is within seed scatter.
- Excellent methodological self-criticism: they note `_noise` (std of successive reward
  changes) **conflates "smooth" with "stuck"**, and name the cleaner metric (std of
  per-token advantages within a rollout).
- **Deduction:** same as Ex1 — the cleaner variance metric is described but not
  implemented.

### Exercise 3 — β vs reward scale · **20/20**
- Prediction correct and tied back to the DPO β-as-implicit-KL-weight lesson.
- Cleanest experiment of the set: swaps to the `!`-counting reward, sweeps β, and — a
  nice touch — **restores the original `reward_fn`** afterward so later cells aren't
  contaminated, with a printed confirmation.
- Result is decisive and correctly interpreted: at β=0.05 the policy hacks the `!`
  reward and **blows past the ~16 positive-word ceiling to ~44**, while β≥0.2 (which
  learned fine under the old reward) now freezes. Same β, different scale, opposite
  behavior — textbook Goodhart. Full marks.

### Exercise 4 — PPO epochs · **19/20**
- Prediction correct on both mechanism (more passes over fixed `old_lp` → ratio drifts
  from 1 → clip engages more) and the staleness/off-policy failure mode.
- Verification confirms the robust effect: **4 → 8 epochs raises `clipfrac` ~15×**
  (≈0.01 → ≈0.15). Honest that the *final-reward* damage is mild/seed-dependent at 60
  iters while the mechanism is unambiguous.
- **Deduction:** the "when does reuse start to hurt" question is answered only
  qualitatively; a KL or reward-variance curve pushed to larger epoch counts (12, 16)
  would have shown the harm turning on, which they had the harness to do.

### Exercise 5 — Kill the critic · **20/20**
- The most impressive answer. The exercise text literally says "replace advantages with
  `returns - returns.mean()`," and the naive reading uses `adv + values` — which **sneaks
  the critic straight back in** via the GAE returns. The group explicitly flagged this
  trap and implemented a **true no-critic**: Monte-Carlo returns via reverse cumulative
  sum (`mc_returns`), mean-centered advantage, **and dropped the value-function loss**.
  That demonstrates real understanding of what the value head is doing, not just a
  mechanical edit.
- Verification is correctly scoped to the **reproducible** claim (no-critic advantage is
  consistently noisier — step-noise ≈1.8 vs ≈1.0) and correctly refuses to over-claim on
  final reward (seed-dependent; ended higher in this run, lower in another). "GAE buys
  variance reduction that compounds over longer training," not a guaranteed higher
  single-run score — exactly the right takeaway. Full marks.

---

## Strengths
- **Methodology followed to the letter** — prediction genuinely precedes each run;
  weights reset for comparability; every claim is checked against printed metrics.
- **Intellectual honesty** is the defining feature: run-dependent orderings, predictions
  that only half-surfaced, and metric limitations are all called out rather than buried.
- **Correct, non-trivial code** — especially the true no-critic in Ex5 and the
  reward-fn restore in Ex3.
- A clear, verified **summary table** that maps back to the notebook's original cheat-sheet.

## What would have earned the last 5 points
1. **Multiple seeds.** Nearly every caveat is "single-seed / run-dependent." Averaging
   3–5 seeds (or plotting mean ± std bands) would have converted the hedged claims
   (Ex1 clipfrac ordering, Ex2 λ=0.95-vs-1, Ex4 final-reward damage) into firm results.
   The group correctly identifies seed noise as the limiter but never spends a run to
   beat it down.
2. **Run the better experiment you named.** Twice (Ex2, Ex5) they propose measuring the
   **std of per-token advantages** as the clean variance metric, and once (Ex1) a
   shorter horizon to expose "faster" — all diagnosed, none executed. Implementing even
   one would have closed the loop from "we know the right measurement" to "we made it."

## Verification note
Cell code was read directly; the quoted metrics live in each exercise's verification
markdown (the large omitted outputs are only the matplotlib figures). Logic in
`_run`, `mc_returns`, `train_nocritic`, and the Ex3 reward swap/restore was checked and
is correct.
