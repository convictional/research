# Evaluating recommender algorithms head-to-head: the methods, in detail

*Recorded 2026-06-25. This is the deep-dive notes file. Goal: catalog the concrete methods for **comparing recommender algorithms/techniques against each other** (algorithm A vs B), how each actually works (data collection, metric calculation, worked examples), what each is good and bad at, who uses it, and what's realistic for us right now. Written plain-English on purpose. Sources are linked inline and collected at the end. Worked numbers are illustrative but computed correctly so the mechanics are clear.*

---

## 1. The big picture: a comparison harness has three layers

Our harness has one job: take two (or more) recommender algorithms and tell us **which is better, with some confidence.** Every method below is really a choice about *three separate layers*. Keeping them separate is the single most useful idea in this whole file:

1. **The yardstick** — *what* you score on. (A relevance metric? Clicks? A human's "was this worth it?" judgment? Estimated retention?)
2. **The procedure** — *how* you actually collect the judgments. (Replay old logs? Show A vs B live? Have humans compare them side-by-side? Ask an AI judge?)
3. **The decider** — the *statistics* that turn a pile of judgments into "A beats B, and we're 95% sure."

Why this matters: most arguments about evaluation are actually about *different layers* and people talk past each other. "Should we use NDCG or human ratings?" is a **layer-1** question. "A/B test or interleaving?" is **layer-2**. "Is this difference real or noise?" is **layer-3**. You mix and match: e.g. *yardstick = human pairwise preference* + *procedure = side-by-side comparison* + *decider = Bradley-Terry model*.

**A few things people call "evaluation" are NOT comparison procedures** — they live outside this harness (Section 3): long-term holdouts (measure cumulative shipped impact, not A-vs-B), behavioral testing (a per-model safety gate), and survey/surrogate work (that *builds a yardstick*, it isn't the comparison itself).

---

## 2. The methods, grouped

Grouped by *where they run*: offline (just logs/data), live (real users), or human-in-the-loop.

---

### GROUP A — Offline comparators (replay data; no live users)

#### A1. Offline metric scoring

**The idea (plain):** take history of what users did, hide the most recent part, let each algorithm produce its recommendations, and score how well those recommendations match what people *actually* went on to do. Higher score = better algorithm.

**How the data is collected / set up:**
- **Split the logs by time** (a "temporal split"): train each algorithm on everything before date T, test on what happened after T. This mimics reality — at serving time you only know the past. *Avoid* random splits or "hide one random item per user" (leave-one-out): they leak the future and can even flip which algorithm looks best ([Ji et al. on data leakage](https://ar5iv.labs.arxiv.org/html/2010.11060)).
- **Ground truth = what the user actually engaged with** in the test period (a click, a watch, a purchase) counts as a "relevant" item.
- **Candidate pool decision (sneaky but important):** do you score the algorithm's ranking against *every* possible item ("full ranking"), or against the true item mixed with a *sample* of random non-items? Sampling is cheaper but **can reverse an A-vs-B verdict** — sampled metrics "do not persist relative statements… not even in expectation" ([Krichene & Rendle, KDD 2020](https://research.google/pubs/on-sampled-metrics-for-item-recommendation/)). Prefer full ranking when you can afford it.

**How the metrics are calculated — one worked example.** Say an algorithm recommends 5 items, and the user actually cared about the items sitting at **rank 2 and rank 4** (everything else was irrelevant). There are 2 relevant items total.

```
Ranks:      1     2     3     4     5
Relevant?   no    YES   no    YES   no
```

- **Precision@5** = (relevant items shown) / (items shown) = 2/5 = **0.40**. *"Of what I showed, how much was good?"*
- **Recall@5** = (relevant shown) / (relevant that exist) = 2/2 = **1.00**. *"Of the good stuff out there, how much did I surface?"*
- **MRR (Mean Reciprocal Rank)** = 1 / (rank of the *first* relevant item) = 1/2 = **0.50**. *"How high was the first good hit?"* — use when only the top result really matters.
- **MAP (Mean Average Precision)** = average of precision *at each relevant position*: precision@2 = 1/2 = 0.5; precision@4 = 2/4 = 0.5; average = **0.50**. Rewards putting relevant items early.
- **NDCG (Normalized Discounted Cumulative Gain)** — the standard for feeds, because it rewards "good stuff near the top" on a sliding scale. Formula: `DCG = Σ relevanceᵢ / log₂(rank+1)`.
  - rank 2: 1 / log₂(3) = 1/1.585 = 0.631
  - rank 4: 1 / log₂(5) = 1/2.322 = 0.431
  - DCG = 0.631 + 0.431 = **1.062**
  - "Ideal" ranking would put both relevant items at ranks 1 & 2: iDCG = 1/log₂(2) + 1/log₂(3) = 1.000 + 0.631 = **1.631**
  - **NDCG = DCG / iDCG = 1.062 / 1.631 = 0.651** (1.0 = perfect ordering).

You compute the chosen metric **per user**, then average across users.

**Beyond-accuracy yardsticks (guard against a feed that's "accurate but awful"):**
- **Intra-list diversity** = average "distance" between the items in one list (using item embeddings). Guards against showing 5 near-identical things.
- **Catalog coverage** = fraction of all items the algorithm *ever* recommends. Low coverage = a few items hog every slot.
- **Novelty** = how non-obvious the items are, usually `-log₂(popularity)`. Guards against "just recommend the most popular thing."
- **Serendipity** = relevant *and* surprising (relevant beyond what a popularity baseline would've shown).
- **Calibration** = does the *mix* of categories in the recommendations match the user's historical mix? Measured with KL-divergence between the two distributions ([Steck, RecSys 2018](https://dl.acm.org/doi/10.1145/3240323.3240372)). Guards against a user's one big interest crowding out everything else.

**How you decide A beats B:** compute the metric per user for both algorithms, then run a **paired significance test** (paired t-test, or Wilcoxon signed-rank if the numbers are skewed) on the per-user differences. "Paired" because the same users are scored under both.

**What it optimizes for:** matching historical behavior, cheaply and automatically.

**Pros:** dirt cheap, instant, fully repeatable, no users needed beyond having some history. Great as a fast regression check ("did my change make things obviously worse?").

**Cons:**
- **Offline ≠ online.** The classic warning: recommending the most *popular* items often scores *best* offline but *worst* with real users ([Garcin 2014, via Castells & Moffat](https://onlinelibrary.wiley.com/doi/full/10.1002/aaai.12051)).
- It only measures "did we reproduce the past," not "is this genuinely good or new."
- **The reproducibility gotcha:** the *same* algorithm gets *different* scores depending on tiny setup choices — split method, how negatives are sampled, k-core filtering, how ties are broken. Documented repeatedly: identical algorithms scored "orders of magnitude" differently across libraries ([Said & Bellogín 2014](https://alansaid.com/publications/2014-said-comparative/)); famous neural models lost to simple baselines once tuned fairly ([Ferrari Dacrema, RecSys 2019](https://arxiv.org/abs/1907.06902)); the *winning* algorithm changes with the filtering threshold ([DaisyRec](https://arxiv.org/abs/2206.10848)). **Implication: pin and version every knob** (split, candidate pool, negative sampling, filtering, tie-breaking, averaging) or your A-vs-B result isn't trustworthy.

**Who uses it:** everyone, as a *first-stage filter* — Netflix and Airbnb both use offline metrics to prune candidates before anything live.

**Our stage:** weak as a *verdict* early (we have little history, and offline-online gap is worst in cold-start), but worth building as a cheap **regression guard** once we have any logs.

---

#### A2. Off-policy / counterfactual evaluation (OPE)

**The idea (plain):** use the logs your *current* feed already produced to *estimate* how a *different* feed would have performed — without shipping it. Like guessing how a new menu would sell using sales records from the old menu.

**How is this different from A1? (they're both offline — this trips people up.)** In one line: **A1 asks "does this algorithm *match what users already did*?" A2 asks "what would this algorithm actually *cause* if we switched to it?"** A1 is a *similarity-to-the-past* test; A2 is a *what-if estimate of the real outcome.*

| | **A1 — offline metric scoring** | **A2 — off-policy evaluation** |
|---|---|---|
| Question it answers | Does B rank the items people historically engaged with near the top? | If B had actually *been* the live feed, what reward (clicks/engagement) would it have earned? |
| What the number means | a *match-the-history* score (e.g. NDCG 0.65) — a proxy, no real-world units | an *estimate of a live metric* (e.g. the avg-reward 1.0 in the worked example below) — the same units an A/B test reports |
| Does it care *how* the logs were generated? | **No** — it treats "what users engaged with" as fixed ground truth | **Yes** — it corrects for the fact that the old feed *chose what users even got to see* (that's what the propensities are for) |
| Mindset | a school exam: "how well does B match the answer key?" (supervised-accuracy thinking) | a what-if: "what would B *cause* if we switched to it?" (causal-inference thinking) |
| Extra data it needs | just (item shown, item engaged) | also the **propensity** = the odds the old feed had of showing each item |

**The deep reason they differ — the "you only clicked what you were shown" trap.** The history A1 grades against was itself produced by whatever feed was running, and people could only engage with what that feed *chose to show them*. A1 ignores this entirely — it just rewards a new algorithm for putting the already-shown-and-clicked items near the top. So A1 is quietly measuring *"how well does B reproduce the old feed's behavior?"*, and it **literally cannot reward a genuinely great recommendation the old feed never showed** (nobody ever had the chance to click it, so it looks "irrelevant"). A2 was invented to escape this trap: by using the propensities, it estimates the value of *B's own choices*, not B's overlap with history. (A2 has a cousin of this limit — if the old feed *never* showed an item at all, A2 can't judge it either; that's the "deficient support" problem noted below, and the reason we'd want to bake in some randomized exposure.)

**They can actually disagree — and that's the whole point.** Remember the "just recommend the most popular items" warning from A1? That algorithm scores *great* on A1 (popular items are exactly what filled the click logs) yet loses with real users. A1 is precisely the method that gets fooled there; A2 is the offline method *built to estimate the reality side*. When the two disagree, A2 is the one closer to what a live A/B test would say — which is why A1 is best used as a cheap sanity/regression check, and A2 as the more honest (but more demanding) estimate of real online value.

**How the data must be collected (the make-or-break part):** every logged event needs four things — the **context** (who/when), the **action** (which item the feed showed), the **reward** (did they engage), and the **propensity** = *the probability the live feed had of picking that item at that moment.*

- **Why propensity is mandatory:** the math reweights each event by `1 / propensity` to undo the old feed's bias. No recorded propensity → no reweighting → the method is simply uncomputable.
- **Why a rigid feed breaks it:** if the feed *always* shows its #1 pick (deterministic), every propensity is 1.0 for the shown item and 0 for everything else — so you have *zero* information about items it never showed. You **cannot retrofit this**; the feed has to deliberately mix in some exploration and **log the odds at decision time.**

**Worked example (IPS = Inverse Propensity Scoring).** Two possible items. The old feed picked randomly (propensity = 0.5 each). We want to estimate a *new* policy that "always shows item 1." Formula: `value ≈ (1/N) Σ rewardᵢ × [1 if new-policy's pick == logged action] / propensityᵢ`.

```
Event  Logged action   Reward   Propensity   New policy picks item1?   Contribution
  1       item 1          1         0.5              match              1/0.5 = 2
  2       item 2          0         0.5            no match                 0
  3       item 1          1         0.5              match              1/0.5 = 2
  4       item 2          1         0.5            no match                 0
IPS estimate = (2 + 0 + 2 + 0) / 4 = 1.0
```

Read it like this: the old feed only chose item 1 *half* the time, so each observed item-1 outcome "stands in for two events" (the ×2 from `1/0.5`). The estimate (1.0) matches the obvious truth that item 1 always paid off here.

- **The weakness — variance:** if event 3's reward had been 0, the estimate would swing to 0.5. Few matching events + big weights = jumpy estimates.
- **SNIPS (Self-Normalized IPS)** divides by the *sum of the weights* instead of by N (here 4/(2+2)=1.0). It's steadier when weights vary a lot ([Swaminathan & Joachims, NeurIPS 2015](https://papers.nips.cc/paper/5748-the-self-normalized-estimator-for-counterfactual-learning)).
- **Doubly Robust (DR)** adds a "reward model" (a guess `q̂` of the reward for any item) as a baseline, and only uses the risky reweighting to *correct* that guess: `DR = (1/N) Σ [ q̂(new pick) + match/propensity × (reward − q̂(logged action)) ]`. It's right if *either* the model *or* the propensities are good — hence "doubly robust" — and it uses every event (lower variance). It's the modern default ([Dudík, Langford, Li, ICML 2011](https://arxiv.org/abs/1103.4601)).

**"Replay" (a simpler cousin):** walk the logs; keep an event *only if* the new algorithm would have picked the same item the log shows; average the rewards of the kept events. **Unbiased — but only if the old feed logged items uniformly at random**, and you throw away a fraction `1/K` of your data (K = number of items), so it's data-hungry ([Li et al., WSDM 2011](https://arxiv.org/abs/1003.5956)).

**How you decide A beats B:** estimate each algorithm's value (with a bootstrap confidence interval) and compare. **Failure to watch for — "deficient support":** if a candidate wants to show items the old feed *never* explored, OPE can't see them and the estimate becomes biased, not just noisy ([Sachdeva et al., KDD 2020](https://www.cs.cornell.edu/~tj/publications/sachdeva_etal_20a.pdf)).

**What it optimizes for:** estimating *real online value* of many candidate algorithms, cheaply, before risking users.

**Pros:** lets you compare lots of candidate feeds offline using real outcomes (not just "did it match history"). **Cons:** needs the propensity logging + exploration built in from day one; high variance / bias when candidates differ a lot from the old feed.

**Who uses it:** Spotify's home-screen bandit ("BaRT") uses exactly this (IPS + normalized variants) and reported its *offline* estimates agreed with *online* results ([McInerney et al., RecSys 2018](https://dl.acm.org/doi/10.1145/3240323.3240354)); Criteo ([Gilotte et al., WSDM 2018](https://arxiv.org/abs/1801.07030)) and ZOZO's open Open Bandit Pipeline ([Saito et al.](https://arxiv.org/abs/2008.07146)) are reference implementations.

**Our stage:** powerful *later*, but the only thing to do *now* is the prerequisite — **if there's any chance we'll want this, log selection odds + add a little randomness from day one.** Can't be added retroactively.

---

#### A3. Simulation

**The idea (plain):** build a model of fake users and let each algorithm interact with it, like a flight simulator. No real users needed.

**How it works:** you author the pieces — a *user model* (interests, how they get bored), an *item model*, a *choice model* (given a list, what does the user click?), and a *transition model* (how the user changes after seeing recommendations). Google's **RecSim** is a toolkit for exactly this decomposition ([RecSim](https://arxiv.org/abs/1909.04847)). A newer variant uses **LLMs as the fake users** — "Agent4Rec" ran 1,000 AI users and reproduced real effects like filter bubbles, for ~$16 of model calls ([Agent4Rec](https://arxiv.org/abs/2310.10108)).

**What it optimizes for:** comparing algorithms safely under explicit, controllable assumptions.

**Pros:** zero real users; can stress-test wild ideas; reproducible. **Cons:** the verdict only reflects the assumptions *you* baked in (you can fool yourself — the "sim-to-real gap"); and the LLM-user version needs the AI to already "know" your items, which it won't for our novel actions.

**Our stage:** useful as a **hypothesis-tester / idea filter** ("does B even behave sensibly?"), *not* as proof. The LLM-user flavor is the published cousin of the "alignsim" idea, but weak for our novel catalog.

---

### GROUP B — Live comparators (need real traffic)

#### B1. A/B testing (online controlled experiments)

**The idea (plain):** randomly split users; half get algorithm A, half get B; measure who does better on a metric you committed to in advance.

**How it works / data collection:**
- **Assignment:** hash each user's ID into a bucket (e.g. `hash(user_id) % 1000`) so the same user always gets the same side, with no stored table. Log "this user was exposed to B" at the moment it happens.
- **Pick your yardstick up front:** one **OEC** ("overall evaluation criterion" — the success metric) plus **guardrail metrics** that must not get worse (latency, unsubscribe/hide rate). Picking the OEC is the hard part: optimizing a shallow proxy like raw clicks famously backfires (clickbait), so the OEC should be a per-user *success/value* measure.
- **Sample size:** the number of users you need scales like `variance / effect²` — **to detect an effect half as big you need 4× the users** ([Kohavi](https://exp-platform.com/Documents/2015-08OnlineControlledExperimentsKDDKeynoteNR.pdf)). This is why A/B is brutal at cold-start.

**Trust checks (cheap, copy these regardless of scale):**
- **SRM (Sample Ratio Mismatch):** a one-line chi-square test that your 50/50 split actually came out ~50/50. If it fails (p < 0.0005), *something is broken* and the results are invalid. Microsoft runs it on **every** experiment; ~6% fail ([Fabijan et al., KDD 2019](https://exp-platform.com/Documents/2019_KDDFabijanGupchupFuptaOmhoverVermeerDmitriev.pdf)).
- **A/A test:** run A against an identical A. If it shows a "winner," your harness/stats are broken, not your feature.

**Efficiency tricks (implementable):**
- **CUPED:** use each user's *pre-experiment* behavior to subtract out noise; reported to cut needed users by ~half ([Deng et al., WSDM 2013](https://dl.acm.org/doi/10.1145/2433396.2433413)).
- **Sequential / "always-valid" testing:** lets you peek and stop early *safely*. (Naively peeking and stopping at the first "significant" moment inflates your false-positive rate ~5× — [Johari et al.](https://arxiv.org/pdf/1512.04922).)

**Watch out for:** novelty effects (a new thing looks exciting then fades — week-1 numbers lie), and network/interference effects (if our early users all interact, one person's treatment leaks to another, breaking the clean split — needs cluster randomization, [Ugander et al.](https://arxiv.org/pdf/1305.6979)).

**What it optimizes for:** the real, causal effect on the outcome we actually care about.

**Pros:** the gold-standard verdict; the only method that proves real cause-and-effect. **Cons:** needs lots of users and weeks of time; many ways to fool yourself.

**Who uses it:** literally everyone (Google, Microsoft, Netflix, LinkedIn, Airbnb, Uber, Spotify all run thousands/year).

**Our stage:** the eventual arbiter, but **underpowered until we have real traffic.** Copy the *safety patterns* (SRM, A/A, guardrails) into any harness now; defer the rest.

---

#### B2. Interleaving

**The idea (plain):** instead of two separate user groups, blend *both* algorithms' picks into **one** list shown to **each** user, then see whose picks they actually engage with. Because every user "votes" on both, you cancel the noise of comparing different people — making it dramatically more sensitive.

**How it works — worked example (team-draft interleaving).** Algorithm A ranks `[a1, a2, a3]`, B ranks `[b1, b2, b3]`. Build one list by coin-flipping who picks each round; each algorithm adds its top unused item:

```
Round 1: coin → A first.  A adds a1, then B adds b1   → list: [a1(A), b1(B)]
Round 2: coin → B first.  B adds b2, then A adds a2   → list: [a1(A), b1(B), b2(B), a2(A)]
...
```

Each slot is tagged with which algorithm contributed it, and the coin-flipping guarantees that "at any position, an item is equally likely to have come from A or B" — which removes position bias.

**How it's scored:** when the user engages with an item, credit goes to the algorithm that contributed it. If a user engaged with `a1` and `a2` (both A's), that user **prefers A**. Tally each user's preferred algorithm across many users, then a simple **sign test** says whether A wins overall.

**Why it's great:** Netflix reports it needs **>100× fewer users** than their most sensitive A/B metric to call a winner; Airbnb reports **~50×** ([Netflix](https://netflixtechblog.com/interleaving-in-online-experiments-at-netflix-a04ee392ec55), [Airbnb](https://medium.com/airbnb-engineering/beyond-a-b-test-speeding-up-airbnb-search-ranking-experimentation-through-interleaving-7087afa09c8e)).

**What it optimizes for:** quickly figuring out *which ranker users prefer* (a relative preference, not a business-value number).

**Pros:** by far the most sample-efficient live comparator for ranking — ideal when users are scarce. **Cons:** still needs *some* live traffic; only tells you *preference*, not long-term value; and it **breaks if an algorithm optimizes the whole list as a set** (e.g. deliberately diversifying), because you can't cleanly blend two such lists — Airbnb saw ~18% disagreement with A/B on exactly those rankers ([Airbnb arXiv](https://arxiv.org/abs/2508.00751)). It's also been shown unsuitable for very low-volume settings ([Chapelle et al.](https://www.cs.cornell.edu/~tj/publications/chapelle_etal_12a.pdf)).

**Our stage:** the *first* live method to reach for once we have any traffic — but note the set-optimization caveat, which may bite a "feed of actions" that's curated as a whole.

---

### GROUP C — Human-judgment comparators (our most viable path now)

When you can't run big live tests, you ask knowledgeable humans to judge the outputs. There's a whole **design space** here — the rating *procedure* (layer 2) and the *aggregation* (layer 3) are separate choices.

#### C1. The rating procedures

- **(a) Absolute / pointwise rating.** Show one recommendation, rate it on a rubric (e.g. 1–5). *Data:* one score per item. *Pros:* simple, gives a magnitude, scales to many items. *Cons:* people use the scale differently and drift over time, so agreement is low and cross-system comparison is noisy.
- **(b) Pairwise / side-by-side.** Show A's output and B's output, ask "which is better, and why?" *Protocol that matters:* **randomize which side is left/right** (kills position bias), keep it **blind** (rater doesn't know which is which), and capture **choice + a magnitude (much/slightly better) + a free-text "why."** *Pros:* judging *differences* is far more reliable than absolute scores, and it's more sensitive. *Cons:* gives *direction*, not an absolute number; comparing everything-to-everything grows fast (n² pairs).
- **(c) Best-worst scaling (MaxDiff).** Show a small set (say 4–5 outputs), ask the rater to pick the **best** and the **worst**. *Why it's clever:* one "best+worst" answer implies many pairwise facts at once (best > the other 3, worst < the other 3), so you extract a near-full ranking from *far fewer* judgments — often more reliable than rating scales. *Cons:* items must be comparable in one view; analysis is a bit more involved (count-based, or feed into a Bradley-Terry / multinomial-logit model).
- **(d) Full-set ranking.** Ask the rater to rank a handful of outputs 1st–5th. Rich signal, but humans struggle past ~5–7 items.
- **(e) Multi-axis rubric.** Rate on *two separate questions* — e.g. Google's **"Needs Met"** (does this serve *this* user's intent right now?) vs **"Page Quality"** (is the thing itself good/trustworthy?) ([Google rater guidelines](https://services.google.com/fh/files/misc/hsw-sqrg.pdf)). *Pros:* untangles different failure types — directly useful for our "right action, wrong time" problem. *Cons:* more rater effort per item.

The canonical large-scale human harness is Google's: ~16,000 trained raters, a written rubric, **side-by-side comparisons** of "proposed change vs current," and — crucially — **the ratings grade the algorithm, they don't train it** ([how-search-works](https://www.google.com/intl/en_us/search/howsearchworks/how-search-works/rigorous-testing/)). The side-by-side methodology is fully described in the open literature ([Thomas & Hawking, CIKM 2006](https://david-hawking.net/pubs/cikmfp633-thomas.pdf)).

#### C2. Aggregation — turning judgments into a winner (layer 3)

- **Don't just average 1–5 scores.** Those are *ordinal* (the gap between 4 and 5 isn't necessarily the gap between 1 and 2), so means are statistically shaky.
- **Bradley-Terry (the right tool for pairwise/best-worst).** It models `P(A beats B) = sigmoid(strengthA − strengthB)` and fits each system a "strength" number. *How you fit it (plainly):* start by assuming every system is equally strong, then keep adjusting: **if a system won more games than its current strength predicts, nudge its strength up; if it won fewer, nudge it down** — stopping once the predictions match what actually happened. For example, equal strengths predict A and B split their 10 games 5–5, but A really won 7, so you raise A's strength (and lower B's) until the model predicts A winning 7 — now it matches reality. Do that across all the matchups together and you get the strengths below. (Mathematically this is just **logistic regression**, so a standard solver runs the loop for you.)

  *Worked example:* 3 systems, comparisons → A beat B 7/10, A beat C 8/10, B beat C 6/10. Fitting gives strengths like (illustrative) A = 0.9, B = 0.3, C = 0.0. Then `P(A beats B) = sigmoid(0.9 − 0.3) = sigmoid(0.6) = 0.65`. To get **confidence intervals**, *bootstrap*: resample the comparisons many times, refit, and look at the spread of strengths. Count a **tie as half a win + half a loss.**

  This is exactly what **LMSYS Chatbot Arena** uses to rank LLMs from millions of human pairwise votes ([LMSYS](https://lmsys.org/blog/2023-12-07-leaderboard/)). The clean **ship rule** falls out: ship B over A if B's strength confidence interval sits above A's.
- **Elo** (the chess system) is the *online, order-dependent* cousin — over-weights recent games; fine for a live leaderboard, worse for a fixed offline comparison. **TrueSkill** adds per-item uncertainty if you need it ([Herbrich et al., NIPS 2006]). For our static "compare these candidates" job, **Bradley-Terry + bootstrap is the best fit.**
- **Rater-quality models (don't trust all raters equally).** **Dawid-Skene** estimates each rater's reliability (their "confusion matrix") *and* the true answer at the same time, then **down-weights** unreliable raters automatically — strictly better than majority vote when some raters are noisy ([Dawid & Skene 1979](https://crowdsourcing-class.org/readings/downloads/ml/EM.pdf)). The open **crowd-kit** library implements Dawid-Skene, Bradley-Terry, and friends ([crowd-kit](https://github.com/Toloka/crowd-kit)) — useful as a *reference for the math* even though we'd implement our own.

#### C3. Quality controls (make human judgments trustworthy)

- **Gold / trap questions:** slip in items where the right answer is known and agreed; raters who miss them get flagged or down-weighted.
- **Attention / time checks:** if someone rates faster than humanly possible (a rough heuristic floor is ~2 seconds per item), down-weight them. (We already track dwell-time — this turns it from a log into a *reliability weight*.)
- **Inter-rater reliability** (how much raters agree — your sanity check that the task is even well-defined): **Cohen's kappa** (two raters, chance-corrected), **weighted kappa** (use this for ordinal 1–5 scales, so "off by one" is penalized less than "off by three"), **Krippendorff's alpha** (handles the realistic case where not everyone rates everything). Rough bands (Landis-Koch): 0.2–0.4 fair, 0.4–0.6 moderate, 0.6–0.8 substantial.

**What this group optimizes for:** *perceived quality* — capturing whether a human expert thinks the recommendation is actually good, which behavior metrics miss.

**Pros:** works with few/no users; we fully control it; pairwise+Bradley-Terry is well-understood and gives a ranking with error bars. **Cons:** human time is the bottleneck; subjective; and (the open problem) judging a recommendation *after the fact* struggles to recreate the *timing/context* that made it good or bad.

**Our stage:** **this is our most viable path right now.** The main design decisions are which procedure (pairwise and best-worst are the strong, efficient options) and committing to Bradley-Terry-style aggregation with reliability weighting.

---

### GROUP D — Scaling human judgment

#### D1. LLM-as-judge

**The idea (plain):** have a strong AI do the same side-by-side comparison a human would, so you can judge thousands of cases cheaply.

**How it works / procedure:** give the model A's output, B's output, and your rubric; ask which is better. **Mitigate the known biases:**
- **Position bias** (it tends to prefer whichever is shown first): **ask twice with the order swapped, and only count a win if it's consistent both ways** (otherwise call it a tie). This is the single most important fix.
- **Verbosity bias** (prefers longer answers): instruct it to ignore length, or normalize.
- **Self-enhancement bias** (prefers its own outputs): don't judge a model with itself.
- For finer scores, use **probability-weighted scoring** (G-Eval): instead of a bare integer, weight the score by the model's token probabilities to get a smooth continuous number ([G-Eval](https://arxiv.org/abs/2303.16634)). Anchor each score level with a written description (Prometheus-style, [Prometheus](https://arxiv.org/abs/2310.08491)).

**Calibrate before trusting it:** run the judge on a set you *also* have human labels for, and check agreement (% agreement vs. the human–human agreement ceiling, plus weighted-kappa / Spearman). In the MT-Bench study, GPT-4 agreed with humans **~85%**, slightly *higher* than humans agreed with each other (~81%) — but that's on general tasks, and judges are weaker on expert-domain calls ([Zheng et al., MT-Bench](https://arxiv.org/abs/2306.05685)).

**What it optimizes for:** scaling the human pairwise method to volume, cheaply.

**Pros:** fast, cheap, decent agreement, lets you compare many candidates often. **Cons:** real biases (need the swap-and-consistency trick), weaker on expert judgments, and it drifts when you change the prompt — so it must be re-validated against humans periodically.

**Who uses it:** ubiquitous in LLM evaluation (MT-Bench, Chatbot Arena's automated tracks). **Our stage:** a strong *throughput multiplier* on top of a human harness — but only after we've calibrated it against our own human labels.

---

## 3. What sits OUTSIDE the comparison harness

These came up in research but are **not** A-vs-B comparison procedures. Worth knowing so we don't mis-file them.

- **Long-term / "global" holdouts — different question.** Keep some users on a frozen *old* experience for months to measure the cumulative value of *everything you've shipped* vs your past self. Answers "are we better than a year ago?", not "which candidate algorithm wins now." It's program-impact measurement (and needs sustained traffic). A long-running A/B fits our paradigm; the holdout *construct* doesn't.
- **Behavioral testing — a gate, not a comparator.** "Many small tests" (does it work for new users? for rare items? stay stable under harmless input changes?) is a *per-model safety/sanity check*, pass/fail. Runs *beside* the comparison as a quality gate; doesn't itself rank A vs B ([RecList](https://arxiv.org/abs/2111.09963)). Still very useful for cold-start (slice tests = cold-start tests), just file it correctly.
- **Survey→learned-proxy & surrogate metrics — these build the *yardstick* (layer 1).** Asking users "was this worth it?" and training a model to predict it (YouTube's "valued watchtime", [YouTube](https://blog.youtube/inside-youtube/on-youtubes-recommendation-system/); Meta's full recipe in ["Retentive Relevance"](https://arxiv.org/pdf/2510.07621)) produces a *metric* you then compare A and B on. Valuable, but it's layer-1 prep, not the comparison itself. (And using the survey answer as a *ranking signal inside the feed* is part of the engine, not evaluation at all.)

---

## 4. What's realistic for *our* stage

Cold-start (few users), small team, building it ourselves, and a *novel* catalog of "recommended actions":

- **Viable now (build around these):** human-judgment comparators — **pairwise or best-worst** procedures, aggregated with **Bradley-Terry + bootstrap confidence intervals**, with reliability weighting and gold/dwell checks. Add an **LLM-judge** to scale throughput once it's calibrated against our human labels. A **multi-axis rubric** (right-for-this-person vs good-in-general) directly attacks our timing/context problem.
- **Cheap to add:** **offline metric scoring** as a regression guard once we have any logs (pin every knob); **behavioral tests** as a side gate (slice tests double as cold-start tests).
- **Build day-one or lose it:** if we ever want **off-policy evaluation**, the feed must **log selection odds + add some exploration from the start.** Cannot be retrofitted.
- **Later, when traffic exists:** **interleaving** first (most sample-efficient — mind the set-optimization caveat), then **A/B** (copy SRM/A-A/guardrails now), then **long-term holdouts** for durable impact.
- **Weak for us specifically:** offline-metric-only verdicts (too little history) and AI-agent simulation (the AI won't know our novel actions).

**The open problem that no method fully solves:** a recommendation's value depends on *timing and context* in a way a search result or a report doesn't ("right action, wrong moment"). Any after-the-fact judgment — human or AI — has to somehow hold "when/why it was shown" fixed, or it measures noise. The multi-axis rubric is a partial handle; this is the genuinely novel part of our problem and worth its own deep dive.

---

## 5. References (grouped, with companies where applicable)

**Offline metrics & reproducibility**
- A Comprehensive Survey of Evaluation Techniques for Recommendation Systems — https://arxiv.org/html/2312.16015v2 — 2023
- On Sampled Metrics for Item Recommendation (Krichene & Rendle, Google) — https://research.google/pubs/on-sampled-metrics-for-item-recommendation/ — KDD 2020
- A Critical Study on Data Leakage in Recommender System Offline Evaluation (Ji et al.) — https://ar5iv.labs.arxiv.org/html/2010.11060 — 2020
- Calibrated Recommendations (Steck, Netflix) — https://dl.acm.org/doi/10.1145/3240323.3240372 — RecSys 2018
- Beyond-accuracy survey (Kaminskas & Bridge) — https://dl.acm.org/doi/10.1145/2926720 — 2017
- Are We Really Making Much Progress? (Ferrari Dacrema et al.) — https://arxiv.org/abs/1907.06902 — RecSys 2019
- Comparative Recommender System Evaluation (Said & Bellogín) — https://alansaid.com/publications/2014-said-comparative/ — RecSys 2014
- DaisyRec-2.0 (hyper-factors that flip results) — https://arxiv.org/abs/2206.10848 — 2022
- Offline recommender evaluation: challenges (Castells & Moffat; contains the Garcin "popular wins offline, loses online" result) — https://onlinelibrary.wiley.com/doi/full/10.1002/aaai.12051 — 2022

**Off-policy / counterfactual (companies: Spotify, Criteo, ZOZO)**
- Doubly Robust Policy Evaluation (Dudík, Langford, Li) — https://arxiv.org/abs/1103.4601 — ICML 2011
- Self-Normalized Estimator / SNIPS (Swaminathan & Joachims) — https://papers.nips.cc/paper/5748-the-self-normalized-estimator-for-counterfactual-learning — NeurIPS 2015
- Unbiased Offline Evaluation / "replay" (Li et al.) — https://arxiv.org/abs/1003.5956 — WSDM 2011
- Off-policy Bandits with Deficient Support (Sachdeva et al.) — https://www.cs.cornell.edu/~tj/publications/sachdeva_etal_20a.pdf — KDD 2020
- Offline A/B Testing for Recommender Systems (Criteo, Gilotte et al.) — https://arxiv.org/abs/1801.07030 — WSDM 2018
- Explore, Exploit, and Explain / BaRT (Spotify) — https://dl.acm.org/doi/10.1145/3240323.3240354 — RecSys 2018
- Open Bandit Dataset & Pipeline (ZOZO) — https://arxiv.org/abs/2008.07146 ; https://github.com/st-tech/zr-obp — 2020

**Simulation (companies: Google)**
- RecSim — https://arxiv.org/abs/1909.04847 — 2019; RecSim NG — https://arxiv.org/abs/2103.08057 — 2021
- RecoGym (Criteo) — https://arxiv.org/abs/1808.00720 — 2018
- Agent4Rec (LLM-agent users) — https://arxiv.org/abs/2310.10108 — SIGIR 2024

**A/B & online experiments (companies: Microsoft, Netflix, Airbnb, LinkedIn, Uber)**
- Online Controlled Experiments (Kohavi, KDD 2015 keynote) — https://exp-platform.com/Documents/2015-08OnlineControlledExperimentsKDDKeynoteNR.pdf — 2015
- CUPED variance reduction (Deng et al.) — https://dl.acm.org/doi/10.1145/2433396.2433413 — WSDM 2013
- Diagnosing Sample Ratio Mismatch (Fabijan et al., Microsoft) — https://exp-platform.com/Documents/2019_KDDFabijanGupchupFuptaOmhoverVermeerDmitriev.pdf — KDD 2019
- Always Valid Inference / peeking (Johari et al.) — https://arxiv.org/pdf/1512.04922 — 2017/2022
- Graph Cluster Randomization (network effects, Ugander et al.) — https://arxiv.org/pdf/1305.6979 — KDD 2013

**Interleaving (companies: Netflix, Airbnb)**
- Interleaving at Netflix — https://netflixtechblog.com/interleaving-in-online-experiments-at-netflix-a04ee392ec55 — 2017
- Interleaving at Airbnb (blog) — https://medium.com/airbnb-engineering/beyond-a-b-test-speeding-up-airbnb-search-ranking-experimentation-through-interleaving-7087afa09c8e — 2022; (paper) https://arxiv.org/abs/2508.00751 — 2025
- Large-scale validation of interleaving (Chapelle et al.) — https://www.cs.cornell.edu/~tj/publications/chapelle_etal_12a.pdf — TOIS 2012

**Human judgment, aggregation & LLM-judge (companies: Google, LMSYS)**
- Google Search Quality Rater Guidelines — https://services.google.com/fh/files/misc/hsw-sqrg.pdf ; How Search Works: rigorous testing — https://www.google.com/intl/en_us/search/howsearchworks/how-search-works/rigorous-testing/
- Evaluation by comparing result sets in context / side-by-side (Thomas & Hawking) — https://david-hawking.net/pubs/cikmfp633-thomas.pdf — CIKM 2006
- Chatbot Arena / Bradley-Terry adoption (LMSYS) — https://lmsys.org/blog/2023-12-07-leaderboard/ — 2023
- Judging LLM-as-a-Judge with MT-Bench (Zheng et al.) — https://arxiv.org/abs/2306.05685 — NeurIPS 2023
- G-Eval — https://arxiv.org/abs/2303.16634 — EMNLP 2023; Prometheus — https://arxiv.org/abs/2310.08491 — 2023
- Dawid-Skene (rater reliability via EM) — https://crowdsourcing-class.org/readings/downloads/ml/EM.pdf — 1979; crowd-kit — https://github.com/Toloka/crowd-kit — 2024
- ResQue user-centric eval questionnaire (Pu, Chen, Hu) — https://www.researchgate.net/profile/Pearl-Pu/publication/221140978_A_user-centric_evaluation_framework_for_recommender_systems/ — RecSys 2011

**Outside the comparison harness**
- RecList (behavioral testing) — https://arxiv.org/abs/2111.09963 — 2022
- YouTube valued-watchtime (survey→model) — https://blog.youtube/inside-youtube/on-youtubes-recommendation-system/ — 2021
- Meta "Retentive Relevance" (full survey→model recipe) — https://arxiv.org/pdf/2510.07621 — RecSys 2025
- Surrogate for Long-Term User Experience (Google) — https://research.google/pubs/surrogate-for-long-term-user-experience-in-recommender-systems/ — KDD 2022
