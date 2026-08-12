# Running graphify on the `app/` directory

The graphify run output and query memories for this run were removed before open-sourcing — they
contained a full extracted graph of a private codebase. The notes below are the findings.

Operationally, basically the same feedback as the first impressions.

Due to the size of the repo, there were about 13k nodes extracted. This is above the 5k limit in the html viz files, so communities were aggregated up in that viz. One thing I noticed is that most communities are labelled with just a number, like "Community X". A generated name or short description per community would be far more useful — an unlabelled community is hard to act on.

I asked 2 queries:
1. what is the architecture of the frontend? at a high level
2. what is the flow of the deep research pipeline?

For the first question, the question could not be answered. The response included points about why the graph is weak here: frontend is a tiny slice of the corpus, and the typescript/JS got chunked alongside the backend code and the frontend contepts ended up scattered across a bunch of communities. Asking Claude Code directly would have answered this well.

For the second question, the response was actually pretty good. That is, I believe it was pretty accurate in the content.

Similar to the data directory exploration, I found the responses to overly reference graph "things". Also, similar to the data directory exploration, token consumption is high for the value returned.

Overall, for these questions the graph-RAG route did not beat asking a coding agent directly against the same code. Direct questioning took a little longer per response, but the responses were consistently more useful.
