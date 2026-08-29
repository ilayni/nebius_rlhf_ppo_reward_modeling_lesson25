# What This Assignment Is About — A Beginner's Guide

## The big picture: teaching a language model to behave

Modern chatbots (ChatGPT, Claude, etc.) are trained in two stages. First they learn
to *predict text* from the internet. Then they're **fine-tuned to behave the way we
want** — helpful, honest, on-topic. That second stage is often done with a technique
called **RLHF** (Reinforcement Learning from Human Feedback), and the workhorse
algorithm inside it is **PPO** (Proximal Policy Optimization).

This assignment is a **"glass-box"** version of that process: instead of calling a
black-box library, everything is built by hand so you can *see every number*. It uses
a tiny model (GPT-2) and a deliberately silly goal so the mechanics stay visible.

## The setup, in plain terms

The lab has four "players":

| Player | What it is | Everyday analogy |
|---|---|---|
| **Actor** (the policy) | GPT-2 — the model being trained | a student learning to write |
| **Critic** (value head) | predicts "how good is this so far?" | a coach guessing the final score mid-sentence |
| **Reference** | a frozen, unchanging copy of GPT-2 | the student's "original habits" |
| **Reward** | a simple rule that scores the output | a teacher grading each essay |

**The silly goal:** the reward simply **counts positive words** ("good", "great",
"love", "amazing"…). The more positive words the model writes, the higher its score.

### What training looks like

The prompt is always: *"I watched the movie last night and…"*

- **Before training**, GPT-2 continues it naturally:
  > "I watched the movie last night and **the director urged me to work swiftly.**"
  (positive-word score: **0**)

- **After training**, the model has learned to chase the reward:
  > "I watched the movie last night and **love love love love love love love…**"
  (positive-word score: **14**)

Notice the model didn't get *smarter* — it learned to **game the score**. That's the
whole point of the lesson: rewards are easy to over-optimize.

## The three "dials" you learn to turn

PPO has knobs. The lab makes three of them tangible:

1. **Clip ε (epsilon) — the "trust region."**
   How big a step the model is allowed to take each update. Too small = learns very
   slowly; too big = takes reckless jumps and destabilizes.

2. **GAE λ (lambda) — the "bias vs. noise" knob.**
   How the coach (critic) estimates how good each word was. One extreme is stable but
   slightly wrong; the other is accurate on average but very noisy.

3. **KL β (beta) — the "leash."**
   How hard we pull the model back toward its original habits (the frozen reference).
   This is the key safety dial:
   - **β = 0 (no leash):** the model runs wild and outputs pure word-salad
     ("great great great love love…") — it **hacks the reward**. This is
     **Goodhart's Law**: *"when a measure becomes a target, it stops being a good
     measure."*
   - **β moderate:** the model improves *and* stays coherent ("it was amazing, really
     beautiful") — the sweet spot.
   - **β too high:** the leash is so tight the model barely changes at all.

## The assignment itself: Section 14

Section 14 is a set of **five experiments**. Each follows the same scientific method:

> **Predict** what will happen → **turn the dial** → **verify** against the results.

The five exercises:

1. **Clip ε** — set it very small vs. very large; watch which learns faster but riskier.
2. **GAE λ** — compare the "stable but biased" setting against the "accurate but noisy" one.
3. **β vs. reward scale** — swap the reward to counting "!" marks (much bigger numbers)
   and discover that the *good* β range **shifts** — a β that worked before now fails.
4. **PPO epochs** — reuse each batch of data more times; watch stability degrade.
5. **Kill the critic** — remove the coach entirely and see how much noisier learning gets.

The **learning goal** isn't the silly positive-word task — it's building intuition for
*how these dials interact*, so that when you later train a real model with a real reward,
you know what each knob does and why RLHF needs a "leash" (the KL penalty) in the first
place.

## The one-sentence takeaway

> This lab shows, with every number visible, **how PPO nudges a language model toward a
> goal — and how easily that goal gets "hacked" if you don't hold the model on a leash.**
