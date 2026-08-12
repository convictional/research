# Context & timing dependence in recommender evaluation ("right item, wrong moment")

*Recorded 2026-06-29. This note digs into the open problem flagged at the end of [03_](03_comparison-methods-in-detail.md) Section 4: a recommendation's value often depends on **when/where/what-mood** it lands in, in a way a search result or a report doesn't. The classic example (Matt's): on Spotify, the same song is great in one mood or time of day and wrong in another. Question: has anyone — academia or industry — studied this, built it into how they *evaluate* recommenders, or do they just not worry about it? Built from a focused 3-stream research pass; written plain-English; sources cited inline and listed at the end. Throughout, I separate **Academic** (theory/lab) from **Industry** (production practice), because the two handle it very differently.*

**A few terms up front (plain):**
- **Recommender / feed** — software that picks what to show you out of too many options.
- **Context** — the *situation* around a recommendation: time of day, day of week, mood, location, who you're with, device, what you're currently doing.
- **CARS — Context-Aware Recommender Systems** — the academic subfield built around the idea that context changes what's a good recommendation.
- **A/B test** — show version A to half of real users, B to the other half, measure which does better live.
- **Off-policy / counterfactual evaluation** — using old logs to estimate how a *new* recommender would have done (see [03_](03_comparison-methods-in-detail.md) A2).
- **In-situ / EMA (Ecological Momentary Assessment)** — asking a user for feedback *right in the moment*, as it happens, instead of afterward.

---

## Q1 — Has anyone looked at "right item, wrong moment"? **Yes — a lot, on both sides.**

### Academic
There's an entire subfield: **Context-Aware Recommender Systems (CARS)**. Its founding idea (Adomavicius & Tuzhilin, *AI Magazine* 2011) is to change the math from "a rating depends on `(user, item)`" to "a rating depends on `(user, item, context)`" — i.e. the *same* user–item pair can be good or bad depending on the situation ([Adomavicius & Tuzhilin 2011](https://ojs.aaai.org/aimagazine/index.php/aimagazine/article/view/2364); updated handbook chapter, [2022](https://link.springer.com/chapter/10.1007/978-1-0716-2197-4_6)). They lay out three ways to use context:
- **Pre-filtering:** first keep only the data matching the situation (e.g. only "evening" listens), then run a normal recommender on it.
- **Post-filtering:** run a normal recommender, then re-rank the output to fit the situation.
- **Contextual modeling:** feed context *into* the model itself (e.g. tensor factorization, [Karatzoglou et al. 2010](https://dl.acm.org/doi/pdf/10.1145/1864708.1864727)).

There's even a research dataset built to prove the point: **LDOS-CoMoDa** tags every movie rating with the viewer's **mood, time of day, companion, and weather**, and *lets the same person rate the same movie differently in different moods* — Matt's Spotify example, made into data ([LDOS-CoMoDa](https://www.lucami.org/en/research/ldos-comoda-dataset/)).

### Industry
Yes — and it's a first-class product idea, not just a research curiosity:
- **Spotify** ships **daylist**, a playlist that literally changes through the day to match "the moments in the day… you usually listen to" particular music ([Spotify Newsroom 2023](https://newsroom.spotify.com/2023-09-12/ever-changing-playlist-daylist-music-for-all-day/)). *(Caveat: Spotify published the product, but no detail on how they evaluate it.)*
- **Netflix** feeds context straight into its recommender: the artwork bandit's inputs include "the device… the time of day and the day of week" ([Netflix TechBlog 2017](https://netflixtechblog.com/artwork-personalization-c589f074ad76)).
- **Uber Eats** ranks restaurants asking "Is it breakfast time or dinner time?… weekday or weekend?" ([Uber 2019](https://www.uber.com/us/en/blog/uber-eats-recommending-marketplace/)).

**Takeaway (Q1):** Both communities clearly recognize the problem. The difference shows up in *how they handle it* — next.

---

## Q2 — Have they built it into *evaluation*? **Academic: yes, explicitly. Industry: mostly indirectly.**

### Academic — context is measured *explicitly*
The academic world bakes context into the evaluation in three concrete ways:
1. **Context-tagged data.** Every interaction in a CARS dataset carries the situation it happened in (the LDOS-CoMoDa mood/time tags above; the **Frappe** app-usage dataset tags each use with daytime/weekday/home-vs-work, [Baltrunas et al. 2015](https://arxiv.org/abs/1505.03014)).
2. **The test asks an in-context question.** Instead of "did you rank a good item highly," it's "**given this exact situation, did you rank the item the user actually engaged with *in that situation* highly?**" Scores (NDCG, Recall@k, etc. — see [03_](03_comparison-methods-in-detail.md) A1) are then computed **per context** (e.g. separately for "evening" vs "morning") or averaged across situations. The open-source CARS toolkit **CARSKit** runs exactly this ([Zheng 2015](https://arxiv.org/abs/1511.03780)).
3. **Time-respecting splits.** For anything time-based, they train on the past and test on the future (a "global temporal split"), never letting the model peek at later events. Random or "hide one item" splits leak the future and are now considered invalid for this ([Ji et al. 2022](https://arxiv.org/pdf/2010.11060); [Klimashevskaia et al., "Time to Split," RecSys 2025](https://arxiv.org/abs/2507.16289)). Session-based recommendation does the in-session version: reveal a session one click at a time and predict the next ([Ludewig & Jannach 2018](https://arxiv.org/pdf/1803.09587)).

### Industry — the moment is captured *implicitly*, by going live
Production teams overwhelmingly do **not** report scores sliced by mood or time of day. Instead:
- **Context becomes a model feature**, and a **live A/B test is the real judge.** Because real users are served *in their real moment*, the live metric (stream rate, watch time) already "prices in" whether the rec fit — a wrong-moment recommendation simply shows up as a skip. Companies say this outright and trust the live test *over* offline scores. Netflix: improved offline accuracy "often" leads to "flat or even negative online metrics" ([Netflix 2024](https://netflixtechblog.com/recommending-for-long-term-member-satisfaction-at-netflix-ac15cada49ef)). YouTube: "for the final determination… we rely on A/B testing via live experiments… live A/B results are not always correlated with offline experiments" ([Covington et al., RecSys 2016](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/)).
- **Explicit context-segmented evaluation is rare**, with a few notable exceptions:
  - **News is the clean exception** — it evaluates time-dependence head-on, because a news article's value visibly decays and the "moment" is just a **timestamp**. The standard benchmark **MIND** splits the data **week-by-week** (train on earlier weeks, test on later), and you replay each click at its real timestamp so "what was fresh then" is built in ([MIND, ACL 2020](https://aclanthology.org/2020.acl-main.331/); unbiased timestamped replay, [Li et al., WSDM 2011](https://dl.acm.org/doi/10.1145/1935826.1935878)).
  - **Spotify's CoSeRNN** research paper *did* break its scores out by time of day and device — and found gains were *largest in unusual contexts* like late-night ([CoSeRNN, RecSys 2020](https://dl.acm.org/doi/10.1145/3383313.3412248)).
  - **Netflix** measures effects separately by device/region/tenure via "heterogeneous treatment effects" ([Netflix 2025](https://netflixtechblog.medium.com/heterogeneous-treatment-effects-at-netflix-da5c3dd58833)) — though not specifically by time of day.
  - **Amazon** ran a test on *when* to recommend at all (only when shopping intent is detected) ([Amazon 2024](https://arxiv.org/html/2404.06017v1)).

**Takeaway (Q2):** Academia has explicit machinery to *measure* context-dependence; industry mostly lets the live experiment absorb it and reports aggregate results. News is where the two meet, because there the moment is cheaply observable (a timestamp).

---

## Q3 — Or do they just not worry about it? **Largely, they route around it — and it's openly unsolved.**

The honest finding, and the reason this is hard: **there is usually no ground-truth label for "the moment" in the data.** You can log *what* a user did, but not *why* — their mood, their intent, "why now" are invisible. You can't compute a score "conditioned on mood" if mood was never recorded. So instead of solving it, the field works around it.

### Academic — names the limit, and offers niche ways to capture the moment
- It's explicitly called out as a core reason offline evaluation is limited: relevance is *situational*, there's no single universal ground truth, and offline logs can't recreate the subjective moment ([Castells & Moffat, *AI Magazine* 2022](https://onlinelibrary.wiley.com/doi/full/10.1002/aaai.12051); the older principle from information-retrieval theory: "relevance cannot be considered without a situation," [Saracevic; Schamber et al.](https://www.sciencedirect.com/science/article/abs/pii/S0306457399000722)). The "accuracy isn't enough" critique is decades old ([McNee, Riedl & Konstan, 2006](https://dl.acm.org/doi/10.1145/1125451.1125659)).
- The niche tradition that *does* attack it directly:
  - **In-situ / EMA** — capture feedback *in the moment* rather than afterward, to avoid the "you can't remember the mood later" problem (a named recsys framework, [IJHCS 2010](https://www.sciencedirect.com/science/article/abs/pii/S1071581910000030); HCI roots, [EMA review](https://pmc.ncbi.nlm.nih.gov/articles/PMC4255457/)). LDOS-CoMoDa does this by recording mood *immediately after* watching.
  - **Scenario-based rater studies** — instead of judging context-free, the human rater is *handed the situation* ("imagine it's Monday morning and you just…") so the judgment is made in-context ([Knijnenburg & Willemsen, UMUAI 2012](https://link.springer.com/article/10.1007/s11257-011-9118-4); modern LLM version, [HELM 2026](https://arxiv.org/html/2601.19197)).
  - Even with logs, the "what if shown at a different moment" question is a **counterfactual** that needs special data (randomized exposure); reconstructing the *unobserved* parts of context is flagged as an open problem ([Jeunen et al. 2023](https://arxiv.org/pdf/2309.04222)).

### Industry — offloads "the moment" to live experiments, and accepts the trade
Production teams largely **do** "just not worry about it" in the post-hoc sense — and it's a deliberate, defensible choice. They let the **live A/B test** carry the moment: real users live their real moments, so the behavioral metric captures fit automatically. The cost they accept is that they lose the **"why"** — a bad-moment recommendation registers as a non-click, with no labeled reason. Reconstructing a *past user's subjective moment* for an after-the-fact judge is essentially treated as infeasible, so nobody tries; they go live instead ([Kohavi, Tang & Xu, *Trustworthy Online Controlled Experiments* 2020]; [Gomez-Uribe & Hunt, Netflix, 2015](https://dl.acm.org/doi/10.1145/2843948)).

**Takeaway (Q3):** It's not that they ignore context — it's that they refuse to *reconstruct* it for a judge. They either go live (real moment, but only behavioral signal) or encode the *observable* slice of context as a model input. The *subjective* moment remains an acknowledged, unsolved gap.

---

## The academic ↔ industry line (why the distinction matters)

- **Academia owns the theory and the explicit measurement tools:** context as a modeled variable, context-tagged datasets, per-context metrics, time-respecting splits. But this works because academic datasets are *small and special* (someone went and collected mood tags). It tells us *what's possible in principle.*
- **Industry owns the practical reality:** at scale, there's no mood label and live traffic is abundant, so the cheap, trustworthy move is to let A/B experiments absorb the moment and skip explicit context-measurement. It tells us *what actually gets done.*
- **They meet in news recommendation**, the one place the "moment" is cheaply observable (a timestamp), so both academia and industry evaluate time-dependence explicitly there.

The practical implication: the theory says "measure context explicitly," but doing so **requires recording the context** — which is a data-collection decision, not just an algorithm choice.

---

## What this means for us (cold-start, building it ourselves)

The industry's #1 answer — **live A/B** — is exactly the one we *can't* lean on yet (too few users), so the moment-problem bites us harder than it bites Spotify or Netflix. The transferable moves, in priority order:
1. **Give the judge the context.** When a human or LLM rates a recommended action, *show them the situation* (when it fired, what was happening in the org, the user's recent activity). This is the academic "scenario-based" fix. It mitigates, doesn't fully solve.
2. **Log the moment with every recommendation** — timestamp, which decisions/events were live, what the user had vs hadn't already seen. This makes the moment *partially observable*, so we can condition on it — our version of the news-timestamp trick, and the prerequisite for *any* explicit context evaluation.
3. **Capture a lightweight in-the-moment signal** ("was this useful right now?") instead of relying only on after-the-fact judgment — the in-situ/EMA idea.
4. **Accept the honest limit:** the user's true subjective state (mood, intent) is unobservable. We condition on the observable slice and acknowledge the rest — which is exactly what everyone else does, too. This is a known limitation, not a failing.

---

## References

*Tagged [A]cademic or [I]ndustry. Dates included.*

**Foundations & theory [A]**
- Adomavicius & Tuzhilin, "Context-Aware Recommender Systems," *AI Magazine* — https://ojs.aaai.org/aimagazine/index.php/aimagazine/article/view/2364 — 2011; handbook update 2022 — https://link.springer.com/chapter/10.1007/978-1-0716-2197-4_6
- Karatzoglou et al., "Multiverse Recommendation" (tensor factorization for context) — https://dl.acm.org/doi/pdf/10.1145/1864708.1864727 — 2010
- Saracevic / Schamber et al., situational relevance ("relevance cannot be considered without a situation") — https://www.sciencedirect.com/science/article/abs/pii/S0306457399000722 — 1990s–2000
- McNee, Riedl & Konstan, "Being Accurate is Not Enough" — https://dl.acm.org/doi/10.1145/1125451.1125659 — 2006

**Datasets with context [A]**
- LDOS-CoMoDa (mood/time/companion-tagged movie ratings) — https://www.lucami.org/en/research/ldos-comoda-dataset/
- Frappe (in-the-wild, context-tagged app usage) — https://arxiv.org/abs/1505.03014 — 2015

**How CARS / sequential models are evaluated [A]**
- Zheng, "A User's Guide to CARSKit" — https://arxiv.org/abs/1511.03780 — 2015
- Ludewig & Jannach, "Evaluation of session-based recommendation algorithms" — https://arxiv.org/pdf/1803.09587 — 2018
- Klimashevskaia et al., "Time to Split" (temporal splitting) — https://arxiv.org/abs/2507.16289 — 2025
- Ji et al., "A Critical Study on Data Leakage in RecSys Offline Evaluation" — https://arxiv.org/pdf/2010.11060 — 2022

**Limits of offline eval / capturing the moment [A]**
- Castells & Moffat, "Offline recommender system evaluation: Challenges and new directions," *AI Magazine* — https://onlinelibrary.wiley.com/doi/full/10.1002/aaai.12051 — 2022
- "In situ evaluation of recommender systems" (IJHCS) — https://www.sciencedirect.com/science/article/abs/pii/S1071581910000030 — 2010
- Ecological Momentary Assessment review — https://pmc.ncbi.nlm.nih.gov/articles/PMC4255457/ — 2014
- Knijnenburg & Willemsen, user-experience evaluation framework, *UMUAI* — https://link.springer.com/article/10.1007/s11257-011-9118-4 — 2012
- Mehta, "HELM" (scenario-based LLM-rec evaluation) — https://arxiv.org/html/2601.19197 — 2026
- Jeunen et al., "Offline RecSys Evaluation under Unobserved Confounding" — https://arxiv.org/pdf/2309.04222 — 2023

**Industry practice [I]**
- Spotify — daylist (product) — https://newsroom.spotify.com/2023-09-12/ever-changing-playlist-daylist-music-for-all-day/ — 2023; "Explore, Exploit, Explain" (BaRT bandit) — https://jamesmc.com/blog/2018/10/1/explore-exploit-explain — 2018; CoSeRNN (context-segmented eval) — https://dl.acm.org/doi/10.1145/3383313.3412248 — 2020
- Netflix — Artwork Personalization (context as bandit input; Replay off-policy) — https://netflixtechblog.com/artwork-personalization-c589f074ad76 — 2017; "Recommending for Long-Term Member Satisfaction" (offline≠online) — https://netflixtechblog.com/recommending-for-long-term-member-satisfaction-at-netflix-ac15cada49ef — 2024; Heterogeneous Treatment Effects — https://netflixtechblog.medium.com/heterogeneous-treatment-effects-at-netflix-da5c3dd58833 — 2025; Gomez-Uribe & Hunt, "The Netflix Recommender System" — https://dl.acm.org/doi/10.1145/2843948 — 2015
- YouTube — Covington et al., "Deep Neural Networks for YouTube Recommendations" (A/B is final arbiter; "example age" freshness feature) — https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/ — 2016
- News — MIND dataset (week-by-week temporal split) — https://aclanthology.org/2020.acl-main.331/ — 2020; Li et al., unbiased offline eval of news bandits — https://dl.acm.org/doi/10.1145/1935826.1935878 — 2011; Das et al., Google News CF — https://dl.acm.org/doi/10.1145/1242572.1242610 — 2007
- Uber Eats — "Recommending for the Marketplace" (meal-time context) — https://www.uber.com/us/en/blog/uber-eats-recommending-marketplace/ — 2019
- DoorDash — Personalized Cuisine Filter (per-daypart bandit) — https://doordash.engineering/2020/01/27/personalized-cuisine-filter/ — 2020
- Amazon — intent-gated recommendation ("when to recommend") — https://arxiv.org/html/2404.06017v1 — 2024
- Jannach et al., "Intent-Aware Recommender Systems" survey (why intent is rarely evaluated explicitly: no ground-truth label) — https://dl.acm.org/doi/full/10.1145/3700890 — 2024
- Kohavi, Tang & Xu, *Trustworthy Online Controlled Experiments* (A/B as gold standard) — 2020
