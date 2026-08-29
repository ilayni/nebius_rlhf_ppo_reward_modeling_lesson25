# Gaps found while converting `ppo_glassbox_llm-final.ipynb` → `.md`

The conversion itself is complete and faithful (all 31 cells, all 14 numbered
sections + subsections carried over 1:1 via `jupyter nbconvert`). The items
below are gaps *in the notebook's own content*, not conversion losses —
flagging them since section 14 (the exercise you were assigned) builds on
material from earlier cells that has these loose ends.

## 1. Section 13 — live slider widget output was never actually captured
The `ipywidgets` cell (section 13, "Optional live sliders") only saved this
truncated placeholder in the notebook JSON:
```
interactive(children=(FloatSlider(value=0.2, description='clip_eps', max=0.6, min=0.05, step=0.05), FloatSlide…
```
This is Jupyter's own truncated `repr()` of the widget, saved as `text/plain`
output — the actual widget state/results from whatever slider values the
author tried were never persisted (no `application/vnd.jupyter.widget-state+json`
metadata in the notebook). So there's no way to see what runs the author did
with the sliders, or what output they produced. If your exercise expects you
to reference "the slider results," they don't exist in this file — you'd need
to run the widget yourself.

## 2. References to a "PPO lecture" / "slides" not included in this file
Section header text repeatedly points outside the notebook:
- Line 3: *"companion to the PPO lecture ... each term from the slides"*
- Line 15: *"everything from the lecture, made turnable"*
- Line 50: *"every quantity from the lecture"*
- Line 67: *"exactly the term in our PPO loss slide"*

None of these lecture slides/deck are embedded or linked in the `.ipynb` —
they're assumed to be a separate course asset (e.g. a Nebius course slide
deck) that this lab is a "companion" to. If you need the exact PPO-loss slide
being referenced, you'll need to pull it from the course materials directly;
it isn't in this repo.

## 3. Reference to a "sentiment PPO demo you'll show next" — not present here
Line 31 (section 3, on the trivial reward function): *"Default rewards
positivity (continuity with the sentiment PPO demo you'll show next)."*
This implies a follow-up/companion notebook or demo that isn't part of this
file and wasn't found elsewhere in this directory as of this conversion.

## 4. Environment note (not a gap, but worth knowing before you run section 14)
The notebook's own recorded run used `device: mps` (Apple Silicon GPU), not
`cuda`, even though the intro text says *"Runs on a single GPU in a few
minutes."* Absolute numbers you'll see in your own re-run (rewards, KL/seq,
clipfrac) may differ slightly from the ones baked into this notebook's saved
outputs depending on your hardware and the random seed's actual behavior on
your device — the qualitative trends (reward ↑, KL bounded vs. runaway per
`beta`) are what the exercises in section 14 are testing, not exact digits.

## No broken/dead links found
There are no `http(s)://` URLs anywhere in the notebook (checked all cell
sources), so there was nothing to verify as broken.
