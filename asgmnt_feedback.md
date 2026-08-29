# Feedback — Section 14 (Student Exercises)

**Submission:** `ppo_glassbox_llm-final_ilay_nisim_boris_kogan.ipynb`
**Authors:** Ilay, Nisim, Boris, Kogan
**Scope graded:** Section 14 only (exercises 1–5). Sections 1–13 are the provided lab
scaffold and were run, not authored.

## Overall

**Excellent.** All five exercises are completed in the exact `predict → turn the dial →
verify` shape the assignment asks for, each with the prediction written *before* the run,
a clean experiment that resets to `policy_fresh_state` for comparability, and a
verification paragraph that checks the outcome against the printed metrics and figures.
The standout quality is **scientific honesty**: where a single-seed run doesn't cleanly
match the prediction, you say so and separate the *robust* effect from the
*seed-dependent* one instead of overclaiming. That is exactly the right instinct.

Grade: **A / distinction-level.** The notes below are refinements, not corrections of
anything wrong.

## Strengths

- **Correct experimental hygiene.** The shared `_run(**kw)` helper resets weights every
  time, so the sweeps are genuinely comparable. In Ex3 you save and restore the global
  `reward_fn` around the `!`-reward override — easy to forget, and forgetting it would
  silently corrupt any later cell.
- **Ex5 goes beyond the prompt, correctly.** The exercise literally says "replace
  advantages with `returns - returns.mean()`," but the hint's `returns` is `adv + values`
  — which smuggles the critic back in. You spotted that, wrote a true Monte-Carlo
  `mc_returns` (reverse cumsum of per-token rewards), *and* dropped the value-function
  loss so it's an honest no-critic baseline. That's the correct reading of the exercise's
  intent, and you explained why in the markdown. Best answer of the five.
- **Right conclusions on the robust effects:** ε=0.6 essentially never clips and drives
  the deepest entropy collapse (Ex1); λ=0 is smooth *because it's biased and barely
  learns* (Ex2); β is scale-dependent and the `!` reward blows past the old ceiling under
  the same β (Ex3, textbook Goodhart); more `ppo_epochs` → ratio drifts further from 1 →
  `clipfrac` rises ~15× (Ex4); no baseline → consistently higher advantage variance (Ex5).
  These are all the intended takeaways.
- **Honest about what a single seed can and can't show** — the 0.05-vs-0.2 clipfrac
  ordering, the 0.95-vs-1.0 noise gap, and final-reward comparisons are all correctly
  flagged as within run-to-run scatter rather than presented as findings.

## Points to tighten (for full rigor)

1. **Single seed is the one real methodological limitation.** You acknowledge it
   everywhere, which is good, but the fix is cheap and would convert several "trend, with
   caveat" verifications into clean ones. Wrapping `_run` in a loop over 3–5
   `torch.manual_seed` values and plotting mean ± std would let you actually resolve the
   0.05-vs-0.2 clipfrac question (Ex1) and the 0.95-vs-1.0 noise question (Ex2) you
   currently have to leave open. If time was the constraint, a one-line note "3 seeds
   would settle this" is enough — but it's the single thing standing between "very good"
   and "airtight."

2. **The `_noise` metric is mildly confounded (Ex2 & Ex5).** It's the std of successive
   reward *changes*, which is a reasonable volatility proxy, but for λ=0 the curve is flat
   largely because the policy is *stuck near the floor*, not purely because the advantage
   variance is low. So "λ=0 is smoothest" partly measures "λ=0 doesn't move." You half-say
   this ("so biased it barely learns"), but the noise number itself inherits the
   confound. A cleaner variance probe would be the **std of the per-token advantage
   estimates within a rollout** (the quantity GAE/critic actually reduce), measured at
   matched reward levels. Same for Ex5 — comparing MC-vs-GAE advantage std directly is a
   more direct read of "what the critic buys" than reward-curve step-noise.

3. **Ex1 "faster learning" half was never cleanly shown.** The prediction had two parts —
   ε=0.6 is *faster* and *riskier*. You verified "riskier" convincingly (entropy collapse)
   but note all three converge to similar reward by 60 iters, so "faster" didn't
   materialize in the reward curve; you pivot to "optimized most aggressively." That pivot
   is defensible, but the honest framing is: *at this reward ceiling and iteration budget,
   the speed difference is masked* — a shorter horizon or a harder reward would surface it.
   Worth one sentence saying so rather than reinterpreting "faster" as "most collapsed."

4. **Minor consistency note.** You standardize on `iters=60` (vs the sweep's original 40)
   and justify it — fine. Just flag that this makes your β numbers not directly comparable
   to Section 12's sweep output, in case a reader lines them up.

## Verdict

Complete, correct, and unusually well-reasoned about the limits of single-seed evidence.
The prediction-first discipline is real (not backfilled), the code is clean, and Ex5 in
particular shows genuine understanding of *why* the baseline matters rather than just
running the prescribed change. Add multi-seed averaging and a variance-specific metric and
this would be publishable-quality lab write-up.
