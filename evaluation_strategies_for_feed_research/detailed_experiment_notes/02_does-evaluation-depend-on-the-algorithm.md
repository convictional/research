# Does our evaluation depend on which algorithm we build?

*Recorded 2026-06-25. A short "learning" note. No new web research — this is reasoning, grounded in the concepts and sources from [01_primer](01_primer-how-recommender-feeds-are-evaluated.md). Written plain-English on purpose.*

---

## The question
If we change *how* the feed works under the hood, do we have to throw out our way of measuring whether it's good? In other words: is the **evaluation strategy** a separate thing from the **algorithm**, or are they tied together?

## Short answer
**Partly separate, partly tied.** The clean way to say it:

> Evaluation is **decoupled** from the algorithm's *internal guts*, but **coupled** to *what the algorithm produces and how it picks/serves things*.

("Decoupled" = independent, can change one without touching the other. "Coupled" = linked, change one and you have to deal with the other.)

## A restaurant analogy
- Judging a restaurant by a **taste test** (is the food good?) doesn't care what **brand of oven** they used. → the taste test is *decoupled* from the kitchen equipment.
- But if you want to grade them by **re-reading their old order receipts**, that only works if they actually *kept records* and *occasionally let customers order at random*. → that method *is* coupled to how they operate.

Same with feeds: judging by "was the recommendation good / did the user act and value it" is implementation-agnostic. The cheaper measurement tricks are not.

## Two layers
1. **The yardstick — what counts as "good" — is algorithm-agnostic.** Anything that looks only at the *outputs* (the recommendations) and *outcomes* (did the user act, were they glad) works the same whether the feed is a simple rule, classic math, a neural net, or an LLM. It treats the algorithm as a sealed box. **This is the part we want to keep stable** so we can compare very different approaches on one fair ruler.
2. **The speed-up tricks are tied to the algorithm.** The cheaper/faster methods depend on what the algorithm emits and how it chooses.

## Where the coupling actually bites
1. **The output type decides which metric even makes sense.** If the algorithm predicts a 1–5 star rating, "how far off was the number" metrics fit. If it ranks a list, you need list-ranking metrics. Different job → different ruler.
2. **Randomness + logging decides whether "estimate from old logs" is even possible.** That trick (called *off-policy evaluation*) is impossible if the feed rigidly always shows its single top pick. It needs the feed to sometimes mix in other options *and* record the odds it used.
3. **Some methods break on some algorithms.** "Interleaving" (blending two algorithms' picks into one list) breaks if an algorithm tunes the *whole list as a set* instead of item-by-item.
4. **Looking *inside* is always algorithm-specific.** E.g., the deep-research variance study could inspect "search terms at depth 0" only because that pipeline had that internal structure to peek at.

## The part people miss: it's a two-way street
It's not just that evaluation depends on the algorithm — **choosing an evaluation method forces requirements back onto the algorithm.** If we want the "estimate from old logs" option later, we have to build the randomness + logging in *now*. So the two get designed together, not in sequence.

## What this means for us
- Keep our **main yardstick algorithm-agnostic** — humans comparing recommendations side-by-side + real outcome signals. That lets us swap feed approaches freely during research without changing the ruler. (Bonus: it's the same approach our deep-research evals already use.)
- Treat the **coupled tricks** (offline proxy metrics, off-policy/old-logs estimation) as **optional add-ons** we pick to match whichever specific algorithm we end up testing.
- **Decide our logging/instrumentation early**, because some evaluation options disappear if we don't capture the right data from the start.

## Where this comes from
The specific couplings above trace to sources already linked in [01_primer](01_primer-how-recommender-feeds-are-evaluated.md): interleaving breaking on set-level optimization ([Airbnb](https://medium.com/airbnb-engineering/beyond-a-b-test-speeding-up-airbnb-search-ranking-experimentation-through-interleaving-7087afa09c8e)); off-policy evaluation needing logged randomness/odds ([Criteo](https://arxiv.org/abs/1801.07030)); and the metric depending on the task/output type ([offline-metrics survey](https://arxiv.org/html/2312.16015v2)).
