# Section 14 — Glass-Box PPO Exercises

**Team:** Ilay, Nisim, Boris, Kogan
**Course:** Nebius RLHF (Session 2 — PPO & Reward Modeling, lesson 25)

Our submission for the Section-14 student exercises of the Glass-Box PPO lab.

---

## What's in this folder

| File | What it is |
|---|---|
| **`ppo_glassbox_llm-final_ilay_nisim_boris_kogan.ipynb`** | **← THE SUBMISSION.** The original lab notebook with a new **"§14 · Answers"** section appended (predictions, code, executed figures, and verifications for all 5 exercises). Fully executed — outputs are embedded. |
| `ppo_glassbox_llm-final.ipynb` | The original, unmodified lab notebook. |
| `ppo_glassbox_llm-final.md` | Word-for-word markdown mirror of the original notebook (for quick reading). |
| `ipynb_gaps.md` | Notes on gaps/loose ends in the original notebook (external lecture refs, unsaved widget output). |
| `asgmnt_plan.md` | The plan we followed to produce the submission. |
| `RL-LLM-Session-2.pdf` | The companion lecture deck the notebook refers to. |

---

## How to run the exercise

### 1. Prerequisites
- **Python with the ML stack.** Use an environment that has `torch`, `transformers`,
  `matplotlib`. On the machine this was built on that's the **Anaconda** Python:
  ```
  $HOME/anaconda3/bin/python     # torch 2.11.0, transformers 5.5.4
  ```
  Verified working versions: torch ≥ 2.1, transformers ≥ 4.3x.
- **Hardware:** runs natively on a Mac — no VM or discrete GPU needed. It auto-selects
  Apple-Silicon GPU (`mps`), CUDA if present, else CPU. Built on an **M4 Pro (48 GB)**;
  each training run (`iters=60`, gpt2) is ~1 minute.
- **Internet (first run only):** the first execution downloads `gpt2` (~525 MB) from
  Hugging Face and caches it under `~/.cache/huggingface/`. Subsequent runs are offline.
  A harmless "unauthenticated requests to HF Hub" warning may appear — no token needed.

### 2. Run it — pick one

**A) In Jupyter (recommended for exploring):**
```bash
cd "<this folder>"
$HOME/anaconda3/bin/jupyter lab      # or: jupyter notebook
```
Open `ppo_glassbox_llm-final_ilay_nisim_boris_kogan.ipynb`, make sure the kernel is the
**Anaconda** Python, then **Run All**. Sections 1–13 build the from-scratch PPO loop;
**§14** runs the five exercises. Full run ≈ 15 min.

**B) Headless (re-execute end-to-end, embed outputs):**
```bash
cd "<this folder>"
$HOME/anaconda3/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  "ppo_glassbox_llm-final_ilay_nisim_boris_kogan.ipynb"
```
`--ExecutePreprocessor.timeout=-1` is required — the training cells exceed nbconvert's
default 30 s per-cell limit.

### 3. What §14 does (the five exercises)
Each is **predict → turn the dial → verify**. Every run resets to the fresh (untrained)
weights `policy_fresh_state` so runs are comparable.
1. **Clip ε** — sweep `clip_eps ∈ {0.05, 0.2, 0.6}`; watch `clipfrac` & entropy collapse.
2. **GAE λ** — sweep `gae_lambda ∈ {0, 0.95, 1}`; watch reward-curve noise.
3. **β vs reward scale** — swap `reward_fn` to `!`-counting; re-sweep β; the good-β range shifts.
4. **PPO epochs** — `ppo_epochs 4 → 8`; `clipfrac` rises, stale reuse hurts.
5. **Kill the critic** — replace GAE advantage with mean-centered Monte-Carlo returns
   (no value baseline, no value loss); advantages get noisier.

### 4. Tuning knobs
Edit the config cell (**§1**, `PPOConfig`) or the per-experiment overrides in §14.
`iters` controls run length (we use 60). The exercises grade on **trends**, not exact
numbers — absolute values are single-seed on `mps` and vary run to run.

---

## Note on the gpt2 cache
The `gpt2` weights (~525 MB) are cached under `~/.cache/huggingface/hub/models--gpt2`.
It is **kept for now** (so the notebook can be re-run during grading). To reclaim the
disk later — only once you no longer need to re-run — delete it with:
```bash
rm -rf ~/.cache/huggingface/hub/models--gpt2
```
The **notebook outputs are already embedded**, so nothing is lost for reading/grading;
the model re-downloads automatically the first time you re-run the notebook afterward.
