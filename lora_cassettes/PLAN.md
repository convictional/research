# POC Plan — LoRA-Adapter Retrieval for Convictional

## 0) One‑liner

Evaluate a **LoRA‑adapter–tuned encoder used only for retrieval** (not generation) over Convictional’s corpus, comparing against current baselines. Focus areas: (a) adapter **selection**, (b) **episodic fine‑tuning**, and (c) **retrieval** incl. adapter routing & loading.

---

## 1) Objective & Success Criteria

**Objective:** Improve retrieval quality on “hard” queries (alias‑heavy, cross‑source, cross‑team) without sacrificing provenance.

**Primary success criteria** (vs. best existing method):

* **Recall@k**: +5–10% at k∈{5,10} on the HEE (high‑entropy entity) query set.
* **nDCG@k**: +3–5% at k∈{10,20}.
* **Freshness recall** on content <30 days old: +5%.
* **Latency** (two‑stage path): P95 ≤ 900 ms end‑to‑end.

**Stop criteria:** <2% improvement after two training episodes or regression on Stability metric.

---

## 2) Scope (what’s in/out)

**In:**

* Single base encoder (frozen) + **tenant‑specific LoRA adapters** trained **unsupervised** on Convictional corpus slices.
* Sources: docs, email, meeting transcripts, Slack, code. (Pair‑mining differs by source.)
* Two retrieval paths: **two‑stage re‑rank** (default) and **multi‑index + RRF** (for ablation).

**Out (for this POC):** graph construction, cross‑encoder rerankers beyond a simple optional layer, knowledge editing, semantic chunking research.

---

## 3) Method Overview

1. **Selection**: choose which adapter(s) to apply per query using a lightweight **router**.
2. **Episodic fine‑tuning**: train/refresh adapters periodically with LoRA on unlabeled text (contrastive/denoising), keeping small **replay** for stability.
3. **Retrieval**: use adapters only to compute/query embeddings; return **chunks + citations**, never answers from weights.

---

## 4) Data & Sampling

* **Chunking** (default): 512 tokens, 64–96 token stride overlap; code uses function‑ or class‑level units when available.
* **Corpus split**: initial snapshot (Episode 0), incremental new/updated docs per episode.
* **Dev sets**:

  * **Stability set**: ~2k chunks representative across sources; never retrained on.
  * **Replay set**: 1–2% of prior episode’s training pairs.
* **HEE query set**: existing “hard” queries curated by team (entity aliases, acronyms, codes); add 10–20 new spanning Slack/email/code.

---

## 5) Positive/Negative Pair Mining (unsupervised)

**Documents**: adjacent sections; same heading or anchor pairs.

**Email**: subject↔body; same thread pairs; consecutive replies.

**Meetings**: Q↔A turns; consecutive turns within ≤90s.

**Slack**: parent↔reply; same thread window; message↔attached file.

**Code**: symbol↔docstring; header↔implementation; callsite↔definition (tree‑sitter).

**Negatives**: in‑batch; **hard negatives** from BM25; periodic mining from current dense index (ANCE‑style).

Optional shaping: **HyDE** pseudo‑queries for each chunk family; cap at ≤10% of steps to avoid overfitting.

---

## 6) Episodic Fine‑Tuning (PEFT)

**Base encoder**: E5/GTE‑class (384–768d). Frozen.

**LoRA config (start point):** r=16, α=32, dropout=0.05; apply to attention (q,k,v,o) + MLP projections.

**Objectives:**

* **Contrastive** (Contriever‑style): in‑batch negatives + mined hards; temperature 0.05–0.1.
* **RetroMAE‑style** (optional ablation): denoising/MAE for encoders.

**Optimizer**: AdamW, lr 2e‑4 (LoRA params only), wd 0.01, bs 1024 tokens‑equiv / 256 pairs (accumulation ok), steps 10–50k depending on corpus delta.

**Schedule:** cosine decay with warmup 5%.

**Replay**: 1–2% of prior episode’s pairs; EWC‑style regularizer optional (λ small) if drift observed.

**Versioning:** semantic version per adapter: `convx/<domain>-<YYYYqN>-v<M.m.p>`; retain last 3.

**Promotion rule:** promote to “active” if delta Recall@10 ≥ +3% on HEE set and no Stability regression >1%.

---

## 7) Adapter Registry & Metadata

**Storage**: LoRA deltas only (20–80 MB each).

**Registry record (JSON schema):**

```json
{
  "adapter_id": "convx/support-2025q4-v1.0.0",
  "base_model": "e5-base-v2",
  "sources": ["docs","slack","email","meetings","code"],
  "objective": "contrastive",
  "episode": 1,
  "train_span": {"from": "2025-08-01", "to": "2025-10-31"},
  "replay_pct": 0.02,
  "hnsw_index_id": "idx_support_q4",
  "metrics": {"recall@10": 0.62, "ndcg@10": 0.48},
  "stability_delta": -0.004,
  "status": "active",
  "created_by": "ml-pipeline",
  "created_at": "2025-11-02T12:00:00Z"
}
```

---

## 8) Retrieval Pipeline

### 8.1 Router (adapter selection)

**Heuristics v1:**

* **Source hint** (optional user or app): if query mentions code symbols / stack terms → prefer `code-*` adapter.
* **Lexical cues**: presence of SKU/ID patterns (e.g., `[A-Z]{2,}-\d+`) → product/specs adapter.
* **Temporal cue**: query includes dates/“last week/quarter” → prefer time‑slice adapter.

If uncertain, route to top‑2 candidate adapters. Log decisions.

**Classifier v2 (optional):** 1–2 hidden‑layer MLP over TF‑IDF features + regex features; outputs top‑N adapters.

### 8.2 Retrieval paths

**Path A — Two‑stage re‑rank (default)**

1. Retrieve `M` (e.g., 400) from **base** dense index (fast) + BM25.
2. Re‑embed those `M` with selected adapter(s) on the fly; compute cosine sim.
3. Fuse lists via **RRF** (k=60) or z‑score normalize then RRF.
4. (Optional) lightweight rerank (ColBERT or cross‑encoder) on final 200.

**Path B — Multi‑index + RRF (ablation)**

* Maintain per‑adapter HNSW/IVF‑PQ index; query base + adapter index directly; RRF fuse.

### 8.3 Output (to eval harness)

Return ranked list of `(doc_id, chunk_id, start_char, end_char, score, adapter_used, debug_features)`.

---

## 9) Harness Integration

**Method ID:** `dense_lora_adapter_retrieval`.

**Config (YAML):**

```yaml
method: dense_lora_adapter_retrieval
base_model: e5-base-v2
adapters:
  - convx/support-2025q4-v1.0.0
  - convx/specs-2025q4-v1.0.0
retrieval:
  path: two_stage  # or multi_index
  M: 400           # base recall pool
  fuse: RRF
  rerank: none     # or colbert|crossencoder
router:
  mode: heuristics # or classifier
  max_adapters: 2
indexes:
  base_dense: hnsw_e5_base
  bm25: okapi_v1
  adapter_dense: { support: idx_support_q4, specs: idx_specs_q4 }
```

**Adapter loading interface (Python pseudocode):**

```python
class LoRAAdapterRetriever:
    def __init__(self, base_model_id, adapter_registry):
        self.base = load_encoder(base_model_id, device)
        self.registry = adapter_registry

    def route(self, query):
        candidates = heuristic_route(query, self.registry)
        return candidates[:2]

    def encode_with_adapter(self, texts, adapter_id):
        with adapter(self.base, adapter_id):  # context manager swaps LoRA weights
            return self.base.encode(texts, normalize=True, batch_size=128)

    def search(self, query, cfg):
        # Stage 1: base + bm25 recall
        base_hits = dense_index.search(query, topk=cfg.M)
        bm25_hits = bm25.search(query, topk=cfg.M//2)
        pool = unify(base_hits, bm25_hits, topk=cfg.M)
        adapters = self.route(query)
        reranked_lists = []
        for a in adapters:
            doc_texts = [get_chunk_text(h) for h in pool]
            q_emb = self.encode_with_adapter([query], a)[0]
            d_embs = self.encode_with_adapter(doc_texts, a)
            scores = cosine_sim(q_emb, d_embs)
            reranked_lists.append(rank(pool, scores, tag=a))
        fused = rrf_fuse([base_hits] + reranked_lists)
        return topk(fused, 50)
```

---

## 10) Experiment Matrix (minimum)

1. **Baseline**: current best (lexical + pretrained dense).
2. **Adapter Two‑Stage**: base recall → adapter re‑rank (contrastive objective).
3. **Adapter Two‑Stage + HyDE**: add pseudo‑query shaping.
4. **Adapter Multi‑Index + RRF**: direct index fusion.
5. **Objective Ablation**: RetroMAE vs. Contrastive (best of 2 advances).
6. **Router Ablation**: heuristics vs. small classifier.

---

## 11) Metrics & Reporting

* **Primary:** Recall@5/10/20, nDCG@10/20, Eval 0–3 score distribution.
* **Freshness**: Recall@10 on docs <30d.
* **Stability**: ΔRecall@10 on Stability set vs. Episode 0.
* **Forgetting gap**: performance drop on Replay diagnostics.
* **Cost/latency:** P50/P95 latency; GPU minutes per episode; index size deltas.

Weekly report: table of metrics by experiment arm + shortlist of failure cases (missed aliases, wrong time slice).

---

## 12) Compute, Storage, Latency (estimates)

* **Adapter training**: 1–3 GPU‑hours (A10/A100) per episode for 1–3M chunks.
* **Two‑stage latency**: +120–250 ms for adapter re‑embed of top‑M (batching on GPU).
* **Index storage**: two‑stage path adds none; multi‑index adds ~0.75 KB/vector (384d fp16) per adapter.

---

## 13) Risks & Mitigations

* **Adapter drift** → use replay + promotion gate; auto‑rollback on Stability regression.
* **Score scale mismatch across adapters** → z‑score normalize or rely on RRF robustness.
* **Cold start** → base model only until corpus ≥50k chunks; then train first adapter.
* **Right‑to‑be‑forgotten** → vectors deleted immediately; if training contamination suspected, rotate adapter (episode+1).

---

## 14) Timeline & Milestones (2 weeks)

* **Day 1–2**: Pair‑mining pipeline; adapter registry; router heuristics.
* **Day 3–4**: Episode 0 training; build two‑stage retrieval; harness integration.
* **Day 5**: Run Eval v1 (arms 1–3); review.
* **Day 6–7**: Hard‑negative mining; Episode 1; rerun Eval.
* **Day 8–10**: Router ablation; optional RetroMAE arm.
* **Day 11–12**: Multi‑index ablation; latency + cost study.
* **Day 13–14**: Final compare, decision gate, write‑up.

---

## 15) Deliverables

* Adapter(s) `*.safetensors` + registry entries.
* Retrieval module (method ID + config) wired into harness.
* Experiment logs, metrics, and decision write‑up (go/no‑go).

---

## 16) Minimal Training Commands (reference)

**Sentence‑Transformers + PEFT (contrastive):**

```bash
python train_contrastive_lora.py \
  --base_model e5-base-v2 \
  --train_pairs /data/pairs.jsonl \
  --output_adapter convx/support-2025q4-v1.0.0 \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --lr 2e-4 --batch_size 256 --max_steps 20000 \
  --hard_negatives /data/hard_negs.jsonl \
  --replay /data/replay.jsonl
```

**Index build (multi‑index option):**

```bash
build_index \
  --adapter convx/support-2025q4-v1.0.0 \
  --chunks /data/chunks.parquet \
  --index_out idx_support_q4
```

---

## 17) Router Heuristics (v1)

| Signal        | Rule                      | Example                                     |
| ------------- | ------------------------- | ------------------------------------------- |
| Code symbols  | Prefer `code-*` adapter   | `NullPointerException`, `pytest`, `kubectl` |
| ID patterns   | Prefer `specs-*`          | `SKU-1042`, `PRD-123`, `JIRA-5678`          |
| Time phrases  | Prefer time‑slice adapter | “Q3 2025”, “last week”                      |
| Slack style   | Prefer `chat-*`           | `@handle`, `#channel`, emoji patterns       |
| Email threads | Prefer `support-*`        | “Re: Subject”, `FW:`                        |

---

## 18) Decision Gate (Go/No‑Go)

Go if **any** adapter arm delivers ≥ +5% Recall@10 on HEE with ≤ 15% cost/latency overhead; otherwise park the feature and revisit with different objectives or richer pair‑mining.
