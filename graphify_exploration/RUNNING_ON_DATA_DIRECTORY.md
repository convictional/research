# Running graphify on the `data/` directory

The graphify run output and query memories for this run were removed before open-sourcing — they
contained a full extracted graph of a private codebase. The notes below are the findings.

Operationally, basically the same feedback as the first impressions.

I asked 4 queries:
1. give me an overview of the graph
2. in plain english, tell me more about fct_user_daily
3. I want to add a metric to fct_user_daily. From a new source (not yet in the graph). What is the lineage that I would need to consider for a change like this, following patterns in the models?
4. any nuances in the implementation of our GTM funnel metrics?

Overall, the responses were okay. Of course, the responses rely only on the graph. So, if something during that process went wrong, that will (could) also surface in the response.

For Question 3, the response contained inaccurate information. It said that if I wanted to do org segmentation (Convictional vs other orgs), I would have to use `internal_organizations`, which is just wrong. Internal orgs is for filtering out internal orgs. The org segmentation comes from a different seed we use. Also, I feel like just asking Claude this would have caught that.

Also, I found the responses to overly reference graph "things". Like, I just want an answer without it telling me about degrees of separation, how many nodes are connected, community cohesion values, or use other graph terminology.

Finally, token consumption during tool calls is high, whether generating the graph or querying it. It's difficult to put actual numbers to it, but I see that my Claude Code usage for today is very high (> 100 million tokens). Also, the context window seems to be getting eaten up pretty quickly. I don't think there is much delegation to actual agents - I think most query tool calls are done within the top level Claude Code "agent".

So, overall, for this corpus and these questions the graph-mediated route did not earn its cost: the questions could have been answered as well or better by asking a coding agent directly against the same code. That takes longer per answer, but the quality difference was worth it.
