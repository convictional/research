# Platform Costs

**Author:** Adam McCabe

This 'experiment' is really some ad-hoc analysis that runs visualizations and summary stats against our `llmrequests` table in postgres (via BigQuery). We focus on extracting token counts by `object` which groups individual requests into one of our first class models (e.g. Decision Process, Meetings, Threads, etc.). In addition we extract counts of threads, meetings and decisions to understand Recall AI cost and velocity of decision & thread creation.

A high-level summary was written up in an internal doc, which is not public.

Broadly, the goal is to be able to understand what it costs to run the Convictional platform for our customers. A good portion of this analysis looks at the number of tokens for a given feature, for example - the typical decision process costs us $0.01 in inference costs. As of the time of writing (Jan 22, 2025) the unit economics were:

| Object Type | Average Tokens per Object | Average Cost per Object ($USD) |
|------------|----------------------|-------------------|
| Thread Summarization | 31,577 | 0.10 |
| Meetings | 21,301 | 0.08 |
| Search Question | 15,456 | 0.05 |
| Organization jobs (e.g. summarize decisions) | 10,379 | 0.03 |
| User Profiles | 7,007 | 0.03 |
| Decision Process | 2,396 | 0.01 |
| Recommend Collaborators and Email Copy | 1,469 | 0.01 |

Absolute totals, per-object counts, and request volumes have been removed from this
published version — they describe the size of a specific customer base rather than the
cost structure. The per-object figures above are the transferable finding: they let you
estimate what an LLM-heavy collaboration product costs to serve per unit of activity.

As with our other experiments, you can run this from the decide directory with:
```bash
cd experiments
gcloud auth # if not previously authorized with google
make run_experiment ARGS="cost_scaling"
```

After the script completes, you'll find the CSV and plot outputs in `cost_scaling/output`.
