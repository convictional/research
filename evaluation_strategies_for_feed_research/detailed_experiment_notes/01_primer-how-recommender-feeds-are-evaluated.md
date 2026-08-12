# How are social-media / recommender feeds evaluated? (plain-English summary)

*Compiled 2026-06-25. This is a research-log entry: a beginner-friendly tour of how the big companies measure whether a "feed" is good, and what that means for us. Built from general knowledge + web searches across six research streams. Sources (with links) are at the bottom. This is brainstorming notes, not a final design.*

---

## What we're even talking about

A **recommender** (or "feed" / "algorithm") is software that decides what to show you when there's *way* too much to show. TikTok's "For You" page, the Facebook feed, YouTube's "up next" — all recommenders. They pick a short list out of millions of options and put it in front of you.

The Convictional Feed will do the same thing, except instead of videos it surfaces **recommended actions** — e.g. "review this proposal," "you should weigh in on this decision." The research question this doc tackles is narrower: **once you build a feed like that, how do you tell if it's any good?** That's the "evaluation" problem.

Turns out there's a whole industry playbook for this. Here it is, simply.

---

## A few words you'll see a lot (mini-glossary)

- **Metric** — a number you measure to judge quality (e.g. "how many people clicked").
- **North-star metric** — the thing you *actually* care about long-term. For most feeds it's **retention** (do people keep coming back?).
- **Proxy metric** — a quick stand-in you can measure *right now* because the north-star takes months to see. Example: clicks. The danger: people optimize the proxy and it goes sideways (more on this below).
- **Offline evaluation** — grade the algorithm against a saved history of what people did before. Like grading a practice test against an answer key. Cheap, fast, but not real life.
- **Online evaluation** — actually put it in front of real people and watch what happens. Real, but slow and needs lots of users.
- **A/B test** — show version A to half your users and version B to the other half, then see which group did better. The gold standard online — *but it needs a lot of people* to give a trustworthy answer.
- **Cold-start** — the situation where you don't have much data yet (few/no users, few interactions). Like a brand-new restaurant with zero reviews: you can't rank dishes by popularity because nobody's ordered yet. **This is us.**

---

## The pattern *everybody* uses (Meta, YouTube, TikTok, Netflix, X, Spotify)

When you look across all of them, they do basically the same four things:

**1. They score each item with a "value model," not by accuracy.**
For every candidate item, they predict how likely you are to do various things (click, like, reply, finish watching…), give each action a **point weight**, and add it up. Highest score wins. They also *subtract* big points for bad outcomes.

*What actually gets "added up," and how it becomes a list:* for **one** post you sum its weighted action-chances into a single **score**; you do that for **every** candidate post; then you **sort them highest-score-first**, and that sorted order *is* the feed. Say a like is worth 1 point, a reply 5, and a share 10, and the model predicts:

| Post | chance you *like* | chance you *reply* | chance you *share* | score |
|------|------|------|------|------|
| A | 0.50 | 0.10 | 0.02 | 1(0.50) + 5(0.10) + 10(0.02) = **1.20** |
| B | 0.20 | 0.30 | 0.05 | 1(0.20) + 5(0.30) + 10(0.05) = **2.20** |
| C | 0.60 | 0.02 | 0.01 | 1(0.60) + 5(0.02) + 10(0.01) = **0.80** |

Sorted high-to-low, the feed shows **B, then A, then C.** Notice Post C is the one you're *most likely to like* (0.60) — yet it loses, because B is far likelier to earn the heavily-weighted reply and share. That's the whole point of a "value model": the **weights** decide what counts as valuable, so a less-likely-but-more-valuable post can beat a more-likely one (different from just predicting what you're most likely to do).

The clearest real example is **X/Twitter**, which open-sourced its actual scoring weights in 2023:

| User action the model predicts | Points |
|---|---|
| You reply, *and the author replies back* | **+75** |
| You open the profile and engage | **+12** |
| You click in and stay ≥2 min | **+10** |
| You reply | **+13.5** |
| You retweet | **+1.0** |
| You like | **+0.5** |
| You hit "show less"/mute the author | **−74** |
| You **report** the post | **−369** |

(Source: [X's open-sourced "heavy ranker"](https://github.com/twitter/the-algorithm-ml/blob/main/projects/home/recap/README.md).) Notice the huge **negative** weights — a "report" is worth −369, swamping everything positive. That's how they say "do no harm." TikTok and Facebook reportedly use the same shape of formula (those specifics were *leaked*, not officially published — flagged below).

**The lesson for us:** a feed's score is usually a weighted recipe of "good things minus bad things," and the weights are a knob *you choose to reflect your values* — not something the math hands you.

**2. They learned that pure "engagement" metrics get gamed — so they add a human "is this actually good?" survey on top.**
If you optimize for clicks, you get **clickbait**. If you optimize for watch-time, you get mindless doomscrolling. Every platform hit this wall:
- **YouTube** switched from clicks to watch-time in 2012 (which immediately *dropped* views ~20%, but they kept it because it was better for users), then went further: they survey users to rate videos **1–5 stars** and only count 4–5 stars as "**valued** watch-time." Because few people fill out surveys, they **train a model to predict your survey answer** for every video. ([YouTube's own explainer](https://blog.youtube/inside-youtube/on-youtubes-recommendation-system/))
- **Facebook** literally asks "**Is this post worth your time?**" and ranks using the answers. ([Meta](https://about.fb.com/news/2021/04/incorporating-more-feedback-into-news-feed-ranking/))

**The lesson for us:** behavior (did they click the action?) is a *weak* proxy. The quality signal you really want is "did the user act on this and feel it was worth it?" — and you often have to *ask*.

**3. They evaluate in stages — cheap to expensive.**
Cheap offline check → a fast online method called **interleaving** (Netflix blends two algorithms' picks into one list for the same person and sees which picks get watched; it's ~**100× more efficient** than a normal A/B test) → a full A/B test on the real north-star → long-term "holdout" groups. ([Netflix](https://netflixtechblog.com/interleaving-in-online-experiments-at-netflix-a04ee392ec55))

**4. They all eventually learn: offline scores lie.**
The famous example: Netflix once paid **$1,000,000** for an algorithm that won an accuracy contest… and then **never used it**, because accuracy didn't translate to people watching more. ([Netflix](https://netflixtechblog.com/netflix-recommendations-beyond-the-5-stars-part-1-55838468f429)) Another study found that recommending the most *popular* items looks **best** on offline tests but performs **worst** with real users. ([Garcin 2014, summarized here](https://onlinelibrary.wiley.com/doi/full/10.1002/aaai.12051)) And a landmark 2019 paper found that 6 of 7 fancy "state-of-the-art" recommenders were beaten by simple, well-tuned baselines once you checked carefully. ([Ferrari Dacrema 2019](https://arxiv.org/abs/1907.06902))

**The lesson for us:** treat offline numbers as a *smoke detector* (catches obvious disasters), never as the final verdict.

---

## Why this is genuinely hard *for us*: the cold-start problem

Almost everything above assumes you have **lots of users and traffic**. A/B tests need thousands of people to give a clear answer. We don't have that yet — we're the restaurant with no reviews.

The good news: there's a known playbook for evaluating when you *don't* have a crowd. Here it is.

### The cold-start toolkit (what works with few/no users)

- **Human experts judging against a written rubric, comparing two options side-by-side.** This is the single most reliable low-data method. The gold-standard example is **Google's Search Quality Rater Guidelines** — a ~180-page rulebook that ~16,000 trained humans use to *judge* whether search results are good. Crucially, the raters' scores aren't fed into the algorithm; they're used to *grade* it. And they compare results **side-by-side (A vs B)** because "which of these two is better?" is far more reliable than "rate this 1–10 in a vacuum." ([Google guidelines](https://services.google.com/fh/files/misc/hsw-sqrg.pdf))
- **LLM-as-judge** — have a strong AI do that same side-by-side comparison, so you can grade thousands of cases cheaply. It agrees with human judges **>80%** of the time *when comparing pairs*. Caveats: it has biases (it tends to prefer whichever option is shown first, and longer answers), and it's weaker on expert-knowledge tasks — so you keep humans in the loop. ([overview](https://eugeneyan.com/writing/llm-evaluators/))
- **Off-policy evaluation** — a clever trick: estimate how a *new* feed would perform using data your *old* feed already collected. Like guessing how a new menu would sell using reviews from the old menu. **The catch:** it only works if your current feed *occasionally shows things somewhat randomly* and records the odds it picked each item. If your feed is rigidly "always show the top pick," this method is mathematically impossible. So if we ever want this, we have to **build a little randomness + logging in from day one.** ([Criteo](https://arxiv.org/abs/1801.07030))
- **Simulation** — build fake users (increasingly, **AI-powered** fake users) and let them react to your feed, like a flight simulator for the algorithm. One project, "Agent4Rec," ran **1,000 AI users** and they reproduced real-world effects like filter bubbles. The catch: a simulator is only as trustworthy as how realistic your fake users are, and these AI users behave a bit unrealistically (e.g. they almost never give harsh 1-star ratings). So it's a good way to *screen out bad ideas*, not to *prove* something works. ([Agent4Rec](https://arxiv.org/abs/2310.10108)) *(This is the published-research version of the "alignsim" idea Adam has been exploring.)*
- **Pick long-term stand-in signals now, and add a "was this useful?" button.** Some easy-to-measure things (like whether recommendations are diverse, or whether people come back) are known to predict long-term retention. Logging a lightweight per-item "worth it? yes/no" from day one means you can do the YouTube-style "predict the survey" trick later.

---

## The good news: we've basically already built the right tool

Here's the punchline. The cold-start gold standard above — **trained human experts, comparing two outputs side-by-side, against a rubric, with quality checks** — is *exactly* what our **deep-research human-evals app** already does:

- blind **pairwise** comparison (A vs B),
- a **5-point scale**,
- two quality dimensions (information quality + style),
- **dwell-time** checks to catch raters who rushed.

That's not a coincidence — it's the same method the whole field converges on when you can't run big A/B tests. So **we can likely reuse most of that machinery** to evaluate the feed.

## The one hard problem that's genuinely new

There's a catch the original research doc already spotted, and it's the real puzzle:

A **research report** has a fixed quality — it's just as good whether you read it today or next week. A **recommended action does not.** Recommending sunscreen is brilliant in July and useless in December — *same recommendation, totally different value, just because of timing/context.*

So when a human rater looks at a past recommendation and asks "was this good?", their answer depends heavily on a context that's hard to reproduce after the fact. That makes the simple "compare A vs B" approach noisier and potentially misleading for a feed. **Cracking how to evaluate a context-dependent recommendation fairly is the core open problem** — and it matters more than the rating mechanics, which we mostly already have.

---

## Where we could go next (open threads)

1. **Define "a good recommended action."** Write the rater rubric — our version of Google's "Needs Met." This is the highest-value first artifact.
2. **Tackle the context problem.** Figure out how to evaluate a "right action, right *time*" recommendation after the fact without it becoming noise. (The hard, novel part.)
3. **Decide what to log from day one** (a bit of randomness + the odds we picked each item + a "worth it?" signal) so the fancier methods become possible later.
4. **Explore the simulation route** — is an AI-user simulator (Agent4Rec / "alignsim" style) worth standing up as a cheap first screen?

---

## Sources

*Grouped by topic. "Official" = the company's own words; "leaked/reported" = journalism about internal docs (treat as less certain); "paper" = academic. A larger ~100-source list from the underlying research is available if we want a fuller bibliography.*

**The big lessons (offline scores can mislead)**
- Netflix — *Beyond the 5 Stars* (the $1M model they never shipped; "75% of watching comes from recommendations") — https://netflixtechblog.com/netflix-recommendations-beyond-the-5-stars-part-1-55838468f429 — official, 2012
- *Are We Really Making Much Progress?* (fancy models beaten by simple baselines: 18 checked, 7 reproducible, 6 beaten) — https://arxiv.org/abs/1907.06902 — paper, 2019
- *Offline recommender evaluation: challenges* (covers the "popular wins offline, loses online" result) — https://onlinelibrary.wiley.com/doi/full/10.1002/aaai.12051 — paper, 2022

**How the big platforms score & evaluate**
- X/Twitter — open-sourced ranking weights (the +75/−369 table above) — https://github.com/twitter/the-algorithm-ml/blob/main/projects/home/recap/README.md — official, 2023 *(the reply weight is 13.5 in the official file; some community write-ups say 27 — use 13.5)*
- YouTube — *On YouTube's recommendation system* ("valued watch-time," 1–5★ surveys, predicting survey answers) — https://blog.youtube/inside-youtube/on-youtubes-recommendation-system/ — official, 2021
- YouTube — *Deep Neural Networks for YouTube Recommendations* (why they optimized watch-time over clicks) — https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/ — paper, 2016
- Meta/Facebook — *Incorporating more feedback into News Feed* ("is this worth your time?" surveys) — https://about.fb.com/news/2021/04/incorporating-more-feedback-into-news-feed-ranking/ — official, 2021
- Meta/Facebook — the leaked "Meaningful Social Interactions" point system (like=1, reshare=5, comment=30, etc.; changed over time) — https://www.cnn.com/2021/10/27/tech/facebook-papers-meaningful-social-interaction-news-feed-math/index.html — **leaked/reported**, 2021
- Instagram — Adam Mosseri, *Instagram Ranking Explained* — https://about.instagram.com/blog/announcements/instagram-ranking-explained — official, 2023
- TikTok — *How TikTok recommends videos #ForYou* (official: finishing a video counts more than weak signals; diversity guardrail) — https://newsroom.tiktok.com/en-us/how-tiktok-recommends-videos-for-you — official, 2020
- TikTok — the leaked "Algo 101" scoring formula + retention goal — https://www.deeplearning.ai/the-batch/what-makes-tiktok-tick/ — **leaked/reported**, 2021
- Netflix — *Interleaving* (the ~100×-more-efficient online test) — https://netflixtechblog.com/interleaving-in-online-experiments-at-netflix-a04ee392ec55 — official, 2017
- Spotify — *Explore, Exploit, and Explain* (BaRT; estimating a new policy from old logged data) — https://dl.acm.org/doi/10.1145/3240323.3240354 — paper, 2018

**Cold-start & human-rating methods (most relevant to us)**
- Google — *Search Quality Rater Guidelines* (the ~16k-rater rubric model; raters grade, don't train) — https://services.google.com/fh/files/misc/hsw-sqrg.pdf — official
- *LLM-as-judge* overview (pairwise > pointwise; >80% agreement; the biases to watch) — https://eugeneyan.com/writing/llm-evaluators/
- ResQue — a ready-made user questionnaire for recommender quality — https://www.researchgate.net/profile/Pearl-Pu/publication/221140978_A_user-centric_evaluation_framework_for_recommender_systems/ — paper, 2011
- *Surrogate for Long-Term User Experience* (using short-term signals like diversity to predict long-term retention) — https://research.google/pubs/surrogate-for-long-term-user-experience-in-recommender-systems/ — paper, 2022

**"Estimate without a crowd" methods (off-policy + simulation)**
- *Offline A/B Testing for Recommender Systems* (off-policy eval; why you must log a bit of randomness) — https://arxiv.org/abs/1801.07030 — paper, 2018 (Criteo)
- *Unbiased Offline Evaluation… ("replay")* — https://arxiv.org/abs/1003.5956 — paper, 2011
- RecSim — Google's recommender simulator — https://research.google/blog/recsim-a-configurable-simulation-platform-for-recommender-systems/ — 2019
- Agent4Rec — 1,000 AI "users" to test recommenders (and where they're unrealistic) — https://arxiv.org/abs/2310.10108 — paper, 2024

**A/B testing foundations (for later, when we have traffic)**
- Kohavi — *Online Controlled Experiments* (the standard reference for running trustworthy A/B tests) — https://exp-platform.com/Documents/2015-08OnlineControlledExperimentsKDDKeynoteNR.pdf — 2015
- CUPED — a trick that roughly halves the users an A/B test needs — https://dl.acm.org/doi/10.1145/2433396.2433413 — paper, 2013
