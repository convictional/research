# Modaic & "confidence" scores for LLM-as-a-Judge — what's real vs. marketing

*Recorded 2026-06-30. A skeptical deep-dive into a vendor — **Modaic** ([modaic.dev](https://www.modaic.dev)) — that claims an "accurate confidence score" for **LLM-as-a-Judge (LLMaaJ)** (using an LLM to grade/score outputs, with a trustworthy number for "how sure is the judge"). Triggered by their post ["Certainty Is All You Need"](https://www.modaic.dev/blog/certainty-is-all-you-need). Goal: understand exactly how their score is computed, whether it's novel, and whether the accuracy claim holds up. Built from a 3-stream research pass; written plain-English; sources cited inline and listed at the end.*

**One-line verdict:** Modaic is a **real but very early, ~2-person startup** (a16z-Speedrun-backed). They **do disclose their method** — and it's a **standard, recently-published technique they didn't invent** — while their headline "**SOTA-accurate** confidence" claim has **no published evidence and no independent verification.** The *idea* is reasonable and reusable; the *boast* is unsubstantiated.

---

## 1. What Modaic is (briefly)

A seed-stage startup (founded 2025; **a16z Speedrun** cohort SR006; ~2 people; **not** Y Combinator — that's a mix-up with a different company, "Moda"). Their product is **"Arbiters"**: LLM judges that return a decision, its reasoning, and a **confidence score**, sold as a hosted, waitlist-gated API. They have genuine open-source tooling (their `gepa-viz` repo has 400+ stars) but the confidence service itself is closed, and the core is built on *external* tools — Stanford's **DSPy** and the **GEPA** prompt optimizer ([Agrawal et al., ICLR 2026](https://arxiv.org/abs/2507.19457)). ([a16z Speedrun listing](https://speedrun.a16z.com/companies/modaic))

---

## 2. How their confidence score actually works *(the detailed part)*

**The problem they target.** LLM judges are known to be **overconfident** — they'll say "I'm sure" while being wrong. The miscalibration is large: one 2025 study measured **Expected Calibration Error** from ~39% (GPT-4o) up to ~74% for a smaller model ([Tian et al. 2025](https://arxiv.org/abs/2508.06225)). A *trustworthy* confidence number would let you auto-accept the judge's high-confidence calls and route only the shaky ones to a human.

**Step 1 — read the judge's "mental state" with a linear probe.** As an LLM generates text, at each step it builds an internal vector of numbers — its **hidden state** (Modaic calls it the "latent vector") — which it uses to pick the next word. That vector quietly encodes a lot about how sure the model is. Modaic trains a **linear probe** on it: a *tiny* classifier that multiplies each number in the vector by a learned weight, adds them up, and squashes the total into a 0–1 confidence (the same "weighted sum" idea as the value-model in [01_](01_primer-how-recommender-feeds-are-evaluated.md)). In their open-source code it is literally one line — `nn.Linear(hidden → 1)`. The probe runs on a small **4-billion-parameter Qwen** model. ([Modaic confidence docs](https://docs.modaic.dev/docs/arbiters/confidence_estimation); [`modaic-ai/probes` repo](https://github.com/modaic-ai/probes); [model weights](https://huggingface.co/modaic/Qwen3.5-4B-probe))

**Step 2 — where the training labels come from.** To teach the probe what "confident" looks like, you need examples labeled with the *true* confidence. Modaic bootstraps those labels three ways:
- **Self-consistency** — ask the *same* judge the *same* question ~10 times (with a little randomness). Answers the same way every time → high confidence; flip-flops → low. Label = fraction of runs that agree ([Wang et al. 2023](https://arxiv.org/abs/2203.11171)).
- **Cross-consistency ("council")** — ask several *different* frontier models (their config lists GPT-5, Claude Opus-4.6, Gemini-3-pro). Label = fraction that agree with the original judge.
- **Human labels** where available.

**Step 3 — the "Align" loop.** Clicking "Align" **re-fine-tunes the probe on fresh labeled data for your specific task**, so the confidence stays calibrated to whatever you happen to be judging.

**The honest way to see it.** Self-consistency and the council are *accurate-ish but expensive* — they cost 10+ model calls per judgment. Modaic uses them only as a **teacher**, distilling them into the cheap probe so that at run-time you get a confidence estimate from **one fast pass** instead of many calls. (The founder's pitch: *"93ms… not by running a second model to grade the first."*) So the genuine selling point is **speed / cost**, not a new kind of accuracy.

---

## 3. Is the method novel? **Not really.**

Reading a model's hidden state with a probe to gauge confidence or truthfulness is a **well-established research line** — [Azaria & Mitchell 2023](https://arxiv.org/abs/2304.13734), [Burns et al. (CCS) 2023](https://arxiv.org/abs/2212.03827), and [Semantic Entropy Probes 2024](https://arxiv.org/abs/2406.15927). Most tellingly, **Modaic's own repo README says its probe is "based off of this paper from Meta Research"** — [arXiv:2512.22245](https://arxiv.org/abs/2512.22245), a Dec-2025 Meta/FAIR paper ("Calibrating LLM Judges: Linear Probes for Fast and Reliable Uncertainty Estimation") with **no Modaic authors**. So the advertised "breakthrough in mechanistic interpretability" is, by their own admission in code, an implementation of someone else's published method.

---

## 4. Does the "accurate / SOTA" claim hold up? **No evidence.**

Their docs assert *"our confidence estimators are able to hit SOTA performance in both AUROC and ECE for LLM judge style tasks"* — but with **no numbers, no dataset, no comparison table, no paper, no reliability diagram.** (**AUROC** = how well the score separates the judge's right answers from its wrong ones; 1.0 = perfect, 0.5 = coin-flip. **ECE** = how *honest* the number is — do the things it calls "80% confident" actually turn out right ~80% of the time?) The marketing site's stat boxes even render as placeholders ("Confidence precision: 0%"). Two reasons for extra skepticism:

1. **It's trained to imitate the very baselines it claims to beat.** The probe learns from self-consistency + council labels, so it can at best *cheaply approximate* them — a **cost** win, not proof it's *more accurate* than them.
2. **It's white-box only.** "White-box" = you need the model's internal numbers, which you only have if you run the model yourself (open weights, like Qwen). So it **cannot** be applied to a Claude or GPT judge through an API — those expose no hidden states (Claude doesn't even expose token probabilities — [Claude API reference](https://platform.claude.com/docs/en/api/messages)). An outside commenter flagged exactly this.

---

## 5. Has anyone independent checked it? **No.**

No third party has reproduced, benchmarked, or critiqued the claim. The only Hacker News post is a **self-submission with 0 comments** from a brand-new account; nothing on Reddit; no public leaderboard entry (RewardBench / JudgeBench); no paper. *(Fair caveat: the post is days old and the company is tiny, so this is partly recency, not a community thumbs-down.)*

---

## 6. Where it sits in the broader confidence landscape

The standard ways to get a confidence number out of an LLM, for context:
- **Verbalized** — just ask it ("rate your confidence 0–100"); known to be badly overconfident ([Xiong et al. 2024](https://arxiv.org/abs/2306.13063)).
- **Token log-probabilities** — use the model's own word-probabilities; decent for base models but **degraded by RLHF** ([GPT-4 report 2023](https://arxiv.org/abs/2303.08774)) and **unavailable on Claude**.
- **Self-consistency** — sample many answers, measure agreement ([Wang 2023](https://arxiv.org/abs/2203.11171)).
- **Semantic entropy** — cluster answers by meaning, measure spread ([Farquhar et al., *Nature* 2024](https://www.nature.com/articles/s41586-024-07421-0)).
- **Judge panels**, **post-hoc calibration** (temperature scaling), and **conformal prediction** — the last being the only one with a real mathematical coverage *guarantee* ([Angelopoulos & Bates 2021](https://arxiv.org/abs/2107.07511)).

A *genuinely* novel result would combine the hard properties **at once** — label-free **and** a distribution-free guarantee **and** robust to domain shift **and** black-box — and benchmark against the current judge-confidence work ([Trust or Escalate, ICLR 2025](https://arxiv.org/abs/2407.18370); [conformal judge intervals, EMNLP 2025](https://arxiv.org/abs/2509.18658); [Google's calibrated autorater 2025](https://arxiv.org/abs/2510.00263); the [Meta probe](https://arxiv.org/abs/2512.22245)). **Modaic clears none of these:** it needs labels, offers no coverage guarantee, is white-box, and says nothing about distribution shift.

---

## 7. What this means for us

- **The reusable idea (worth keeping):** *distill* an expensive confidence signal (self-consistency, or a panel of judges) into a **cheap probe**, so you get a confidence number in one fast pass instead of 10+ model calls. Useful if we ever build a judge-based eval at volume.
- **The catches:** it's **white-box** (we'd have to self-host an *open* judge model — it won't work on a Claude/GPT judge), and it **needs labels** to train the probe. We could also just build it ourselves straight from the **free Meta paper** — Modaic's value-add is packaging and speed, not secret science.
- **Bottom line:** treat the "accurate / SOTA confidence" claim as **marketing until they publish data**. The genuinely interesting, true part is the cheap-distillation pattern — not a calibration breakthrough.

---

## Sources

**Modaic (primary, all accessed 2026-06-30)**
- "Certainty Is All You Need" (blog) — https://www.modaic.dev/blog/certainty-is-all-you-need
- "How Modaic Measures Confidence" (docs) — https://docs.modaic.dev/docs/arbiters/confidence_estimation
- `modaic-ai/probes` repo (README states it's based on the Meta paper) — https://github.com/modaic-ai/probes ; `probes-v2` — https://github.com/modaic-ai/probes-v2
- Confidence-probe model weights (LoRA adapter on Qwen-4B) — https://huggingface.co/modaic/Qwen3.5-4B-probe
- a16z Speedrun portfolio listing — https://speedrun.a16z.com/companies/modaic

**The method's actual origins / prior art**
- Radharapu et al. (Meta/FAIR), "Calibrating LLM Judges: Linear Probes for Fast and Reliable Uncertainty Estimation" — https://arxiv.org/abs/2512.22245 — Dec 2025 *(the paper Modaic's probe is based on)*
- Azaria & Mitchell, "The Internal State of an LLM Knows When It's Lying" — https://arxiv.org/abs/2304.13734 — 2023
- Burns et al., "Discovering Latent Knowledge in LLMs Without Supervision (CCS)" — https://arxiv.org/abs/2212.03827 — 2022 (ICLR 2023)
- Kossen et al., "Semantic Entropy Probes" — https://arxiv.org/abs/2406.15927 — 2024
- Agrawal et al., "GEPA" (external prompt optimizer Modaic builds on) — https://arxiv.org/abs/2507.19457 — ICLR 2026

**Confidence / calibration landscape & LLM-as-judge**
- Wang et al., "Self-Consistency Improves Chain-of-Thought Reasoning" — https://arxiv.org/abs/2203.11171 — ICLR 2023
- Xiong et al., "Can LLMs Express Their Uncertainty?" (verbalized overconfidence) — https://arxiv.org/abs/2306.13063 — ICLR 2024
- Farquhar, Kossen, Kuhn & Gal, "Detecting hallucinations using semantic entropy," *Nature* 630 — https://www.nature.com/articles/s41586-024-07421-0 — 2024
- Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction" — https://arxiv.org/abs/2107.07511 — 2021
- Zheng et al., "Judging LLM-as-a-Judge with MT-Bench" — https://arxiv.org/abs/2306.05685 — NeurIPS 2023
- Tian et al., "Overconfidence in LLM-as-a-Judge" (ECE numbers) — https://arxiv.org/abs/2508.06225 — 2025
- OpenAI, "GPT-4 Technical Report" (RLHF degrades calibration) — https://arxiv.org/abs/2303.08774 — 2023
- Anthropic Messages API reference (no logprobs exposed) — https://platform.claude.com/docs/en/api/messages

**The "novelty bar" — current judge-confidence work to benchmark against**
- Jung, Brahman & Choi, "Trust or Escalate" (selective judging with a guarantee) — https://arxiv.org/abs/2407.18370 — ICLR 2025
- Sheng et al., "Analyzing Uncertainty of LLM-as-a-Judge: Conformal Intervals" — https://arxiv.org/abs/2509.18658 — EMNLP 2025
- Li et al. (Google), "Judging with Confidence: Calibrating Autoraters to Preference Distributions" — https://arxiv.org/abs/2510.00263 — 2025
