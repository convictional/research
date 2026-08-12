# decisions\_to\_goals

An experiment that evaluates three different **decision→goal mapping schemas** by compressing each mapping into a fixed-length neutral summary, then scoring the summaries with a 9-judge MoE LLM ensemble.

**Models:** Claude Opus 4.7 (mining + judging), Sonnet 4.6 (mapping + summarization + judging), Haiku 4.5 (judging).
**Corpus:** 242 decisions, 1,926 activity events, date-bounded 2025-04-29.
**Judges per cell:** 9 (3 models × 3 personas).
**Temperature:** single temperature (default 0.0); reported, not evaluated as a variable. Some judge models (e.g. Opus) run at their API default since they reject the temperature parameter.
**Scoring scale:** 5 dimensions binary 0/1; overall = sum (0–5); cell score = trimmed mean of 9 judges (drop high + low).

---

## End-to-End Flow

```mermaid
flowchart TD
    subgraph Data["Shared Corpus (once)"]
        D[242 decisions]
        E[1926 activity events]
        SG[Convictional stated goals]
    end

    subgraph P1["Phase 1 — Goal Mining (per condition)"]
        S1[Step 1: Mine unstated goals\nfrom activity corpus]
        S2[Step 2: Validate stated goals\nagainst activity evidence]
        S3[Step 3: Consolidate + deduplicate\nembedding similarity threshold 0.85]
        S4[Step 4: Alignment report\nsynergies and tensions]
        S5[Step 5: Summary\nFinalizedGoalSet]
        S1 --> S3
        S2 --> S3
        S3 --> S4 --> S5
    end

    subgraph P2["Phase 2 — Decision Mapping (3 schemas)"]
        A[Step A: Shared analysis\none per decision]
        DM[DM — Single-goal\neach decision to at most 1 goal]
        DSM[DSM — Scored\neach decision scored 0 to 1\nagainst all goals]
        GM["GM — Graph\nlabeled edges\n8 relation types:\nadvances, blocks, informs, depends_on,\nsynergizes_with, tensions_with,\nsupersedes, is_evidence_of"]
        A --> DM
        A --> DSM
        A --> GM
    end

    subgraph OBF["Phase 2.5 — Obfuscation Layer"]
        RAW["Raw artifacts\n16k to 30k words each\nstructurally different"]
        SUM["Fixed-length summaries\n~450-600 words each\nneutral prose, schema hidden"]
        RAW -->|"LLM summarizer\nSonnet at temp 0\nhard word budget"| SUM
    end

    subgraph P3["Phase 3 — MoE Judge Ensemble"]
        J["9 judges per cell\n3 models x 3 roles\n5-dim binary rubric"]
        R["Trimmed mean overall\nper-dim means\nvariance decomposition"]
        J --> R
    end

    subgraph OUT["Outputs"]
        RM[results_matrix.json/csv]
        RMD[RESULTS.md]
        VIZ[viz/index.html\ninteractive graph]
    end

    Data --> P1
    P1 --> P2
    P2 --> OBF
    OBF --> P3
    P3 --> OUT
```

---

## Goal-Set Conditions

Three conditions vary the **provenance** of the goal set. The decision corpus is identical across all three.

| Condition | Scenario | Step 1 (mine unstated) | Step 2 (validate stated) | Goal set |
|-----------|----------|------------------------|--------------------------|----------|
| `unstated` | Fresh-onboarded company, 0 written goals | run | no-op | unstated only |
| `stated` | Has written goals, no approved unstated corpus | skip | run | stated only |
| `mixed` | Future platform combining both | run | run | both, merged |

```mermaid
flowchart LR
    subgraph unstated["unstated condition"]
        direction TB
        U1["Step 1: Mine unstated ✅"]
        U2["Step 2: Validate stated\nno-op — no stated goals ⏭️"]
        U3["Canonical goals:\nunstated only"]
        U1 --> U3
    end

    subgraph stated["stated condition"]
        direction TB
        T1["Step 1: Mine unstated\nSKIPPED ⏭️"]
        T2["Step 2: Validate stated ✅"]
        T3["Canonical goals:\nstated only"]
        T2 --> T3
    end

    subgraph mixed["mixed condition"]
        direction TB
        M1["Step 1: Mine unstated ✅"]
        M2["Step 2: Validate stated ✅"]
        M3["Canonical goals:\nboth merged + deduped"]
        M1 --> M3
        M2 --> M3
    end
```

---

## Goal Mining Pipeline (Phase 1)

```mermaid
flowchart TD
    IN["Activity events + decisions\n+ stated goals (optional)"]

    S1["Step 1 — Unstated Extraction\nOpus 4.7\nMines implicit goals from\nbehavioral evidence\nOutputs: CandidateGoal list\nSKIPPED for stated condition"]

    S2["Step 2 — Stated Validation\nSonnet 4.6\nValidates each stated goal\nagainst activity evidence\nOutputs: StatedGoalEvidence list\nno-op for unstated condition"]

    S3["Step 3 — Consolidation\nOpenAI text-embedding-3-small\nMerge stated + unstated\nDeduplicate by cosine sim >= 0.85\nAssign stable UUID4 IDs\nOutputs: CanonicalGoal list"]

    S4["Step 4 — Alignment Report\nSonnet 4.6\nSynergies and tensions\nbetween canonical goals\nOutputs: GoalRelation list"]

    S5["Step 5 — Summary\nOpus 4.7\nNarrative markdown summary\nOutputs: FinalizedGoalSet pkl"]

    IN --> S1 --> S3
    IN --> S2 --> S3
    S3 --> S4 --> S5
```

---

## Three Mapping Schemas (Phase 2)

All three schemas share Step A (analysis) to isolate schema-level effects.

```mermaid
flowchart TD
    IN2["Decisions + CanonicalGoals"]

    SA["Step A — Shared Analysis\nSonnet 4.6\nOne LLM call per decision\nOutputs: MappingAnalysis\nwith confidence-tagged assumptions\nCached: mapping_analysis.pkl"]

    DM["DM — Direct Mapping\nSonnet 4.6\nEach decision to at most 1 goal\nor null if none applies\nDMEntry: goal_id, confidence, reasoning"]

    DSM["DSM — Direct Score Mapping\nSonnet 4.6\nEach decision scored 0-1\nagainst every goal\nOnly emit scores >= 0.20\nDSMEntry: scored_goals list"]

    GM["GM — Graph Map\nSonnet 4.6\n8-relation labeled edges\ndecision-to-goal and goal-to-goal\nRelations: advances, blocks, informs,\ndepends_on, synergizes_with, tensions_with,\nsupersedes, is_evidence_of\nNOTE: decision-to-decision edges are\nstructurally impossible — mapper only\nsees one decision per call"]

    RENDER["Render to schema-masked markdown\nIdentical 5-section structure\nForbidden: DM/DSM/GM/Direct Mapping/Graph Map\nOrphan decisions bucketed in\nUnattached / Miscellaneous section"]

    IN2 --> SA
    SA --> DM
    SA --> DSM
    SA --> GM
    DM --> RENDER
    DSM --> RENDER
    GM --> RENDER
```

---

## Obfuscation Layer — Key Diagram

The core fix for the apples-to-oranges judging problem.

**Problem:** GM artifacts are ~29k words, DM/DSM ~17-19k. LLM judges bias toward volume, so GM wins unfairly not because it is a better schema, but because it presents more text.

**Fix:** Compress every artifact to a fixed-length, schema-neutral prose summary before judging.

```mermaid
flowchart LR
    subgraph Raw["Phase 2 Raw Artifacts (different sizes)"]
        RDM["DM: ~17k words\nflat list structure"]
        RDSM["DSM: ~18k words\nscored-goal structure"]
        RGM["GM: ~29k words\ngraph edge structure"]
    end

    subgraph OL["research_summary.py — Obfuscation Layer"]
        SUMR["Sonnet @ temp 0.0\nHard word band: 450-600 words\nFixed 5-section structure:\n  Overall Landscape\n  Strongest Connections\n  Coverage and Orphans\n  Tensions and Synergies\n  Gaps and Cautions"]
        GUARD["Anti-leak guards\n_HARD_FAIL_TOKENS: graph/node/edge/vertex/\n  network/diagram/threshold/relation type,\n  0.NN decimals, bracket tiers [high]/[medium]/[low]\n_WARN_RETRY_TOKENS (retry once):\n  scored/scoring/single-goal/one-to-one"]
    end

    subgraph Sums["Fixed-Length Summaries (same size)"]
        SDM["DM summary\n~525 words\nneutral prose"]
        SDSM["DSM summary\n~525 words\nneutral prose"]
        SGM["GM summary\n~525 words\nneutral prose"]
    end

    subgraph CAL["Calibration Pilot (pre-judge gate)"]
        CA["Check A: all summaries within 450-690 words\nmax/min ratio <= 1.25\nProves volume gap is gone"]
        CB["Check B: padded GM summary +50% filler\ndelta <= 0.2 points\nProves length-bias guard works"]
    end

    RDM -->|summarize| SDM
    RDSM -->|summarize| SDSM
    RGM -->|summarize| SGM
    OL -.->|enforces| SDM
    OL -.->|enforces| SDSM
    OL -.->|enforces| SGM
    SDM & SDSM & SGM --> CA
    SGM --> CB
```

---

## MoE Judge Ensemble (Phase 3)

```mermaid
flowchart TD
    SUM["Fixed-length research summary\n~450-600 words, schema identity hidden"]

    subgraph JUDGES["9 Judges (3 models x 3 roles)"]
        direction LR
        OA["Opus + strategy_analyst"]
        OO["Opus + ops_reviewer"]
        OS["Opus + skeptic"]
        SA["Sonnet + strategy_analyst"]
        SO["Sonnet + ops_reviewer"]
        SS["Sonnet + skeptic"]
        HA["Haiku + strategy_analyst"]
        HO["Haiku + ops_reviewer"]
        HS["Haiku + skeptic"]
    end

    subgraph RUBRIC["5-Dimension Binary Rubric (0/1 each, 0-5 total)"]
        direction LR
        C["Coverage\n~60%+ of decisions have\ndefensible connections"]
        F["Fidelity\nsemantically correct"]
        SQ["Synthesis Quality\ngenuine prioritization vs flat list"]
        I["Interpretability\ngoal owner can follow it"]
        ID["Information Density\nno padding, no starvation"]
    end

    AGG["Trimmed mean overall\ndrop high + low of 9\nper-dim means + variance\nmodel and role decomposition"]

    SUM --> JUDGES
    JUDGES --> RUBRIC
    RUBRIC --> AGG
```

**Anti-bias controls:**
- Cell ID never passed to any judge prompt
- Schema identity is never written into the raw artifact — masked *by construction* in `render.py` (the dm/dsm/gm label lives only in filenames/cache keys; no blacklist is applied). The generated summary is additionally guarded by `_HARD_FAIL_TOKENS` (hard-fail) and `_WARN_RETRY_TOKENS` (warn-and-retry once) in `research_summary.py`
- `information_density` dimension actively penalizes length/repetition
- Calibration pilot checks: (A) cross-schema length normalization, (B) padding-bias delta ≤ 0.2

---

## Orphan Goal Handling

```mermaid
flowchart LR
    subgraph Mapper["LLM Mapper\nnormal decisions"]
        D1["Decision A → Goal X"]
        D2["Decision B → Goal Y"]
    end

    subgraph Orphans["LLM Mapper\norphan decisions"]
        D3["Decision C → null/no edge"]
        D4["Decision D → null/no edge"]
    end

    subgraph Render["render.py (deterministic, no LLM)"]
        ORG["Synthetic Orphan Goal\nid: 000...000\ntitle: Unattached / Miscellaneous Decisions\nis_orphan=True\nnever saved to goal set pkl\ninjected at render time only"]
        D3 --> ORG
        D4 --> ORG
    end

    subgraph Outs["Outputs"]
        MD["mapping_*.md\nSection 1 Summary: Unattached decisions: 2\nIdentical orphan subsection\nacross all 3 schemas"]
        VIZG["viz graph\norphan node grey\nunattached edges dark grey\nn_orphans in cell meta"]
    end

    Render --> MD
    Render --> VIZG
```

**Why visible orphans matter:** Without this, decisions with no goal match disappear — GM drops them entirely. A large orphan bucket signals goal-set coverage gaps and correctly lowers the Coverage dimension score.

---

## CanonicalGoal Data Contract

```mermaid
classDiagram
    class CanonicalGoal {
        +str id
        +str title
        +str description
        +bool is_stated
        +bool is_unstated
        +bool is_orphan
        +list origin_stated_goal_ids
        +list origin_unstated_candidate_ids
        +float activity_support_score
    }
    note for CanonicalGoal "id: UUID4, assigned once at Step 3 consolidation\nStable across Phase 2 mapping and Phase 3 judging\nis_stated XOR is_unstated (enforced by validator)\nis_orphan=True: both is_stated and is_unstated = False\nOrphan id 000...000 never saved to pkl"

    class FinalizedGoalSet {
        +str condition_name
        +list goals
        +AlignmentReport alignment_report
        +str summary_markdown
        +dict run_metadata
    }

    class StatedGoal {
        +str id
        +str title
        +str description
        +str source
    }

    class CandidateGoal {
        +str title
        +str description
        +list supporting_evidence
        +list source_event_ids
    }

    FinalizedGoalSet "1" --> "many" CanonicalGoal : contains
    StatedGoal --> CanonicalGoal : promoted in Step 3
    CandidateGoal --> CanonicalGoal : promoted in Step 3
```

---

## Data Ingress

The experiment's input corpus (decisions, activity events, stated goals) is read
from the **local Postgres DB that `make research_load` populates**, not from a
remote/production database.

**Step 0 — load the research bundle (prerequisite, run once):**

```bash
# From app/web — downloads the latest research export from GCS and loads it into
# the local decide_<env>_decide DB (requires GCP auth). The experiment never
# invokes this for you.
cd app/web && make research_load
```

The experiment then connects automatically on the first data-loading command
(`build_dataset`, `mine_all`, or `run_all`) using `settings.postgres_dsn`
(default `postgresql://127.0.0.1:5432/decide_<env>_decide`; override with
`POSTGRES_DSN` in `.env`). The corpus is cached to `output/shared/*.pkl`, so the
DB is only hit on the first run.

## Run Order

```bash
# Whole experiment end to end (all phases, with a progress bar)
make run_experiment ARGS='decisions_to_goals run_all'
```

Or run each phase individually:

```bash
# Phase 1 — goal mining (once per condition)
make run_experiment ARGS='decisions_to_goals mine_all'

# Phase 2 — decision mapping (3 schemas x 3 conditions = 9 cells)
make run_experiment ARGS='decisions_to_goals map_all'

# Phase 2.5 — obfuscation layer (compress artifacts to fixed-length summaries)
make run_experiment ARGS='decisions_to_goals summarize_all'

# Calibration gate (MUST pass before judging)
make run_experiment ARGS='decisions_to_goals calibration_pilot'

# Phase 3 — MoE judging at a single temperature (default 0.0)
make run_experiment ARGS='decisions_to_goals judge_all --temperature 0.0'

# Assemble results
make run_experiment ARGS='decisions_to_goals build_matrix'
```

**Machine-readable results:** `output/results_matrix.{csv,json}`
**Human-readable summary:** `RESULTS.md`
**Phase 3 judge caches:** `output/{unstated,stated,mixed}/judge_{dm,dsm,gm}_T0.0.pkl`
**Research summaries:** `output/{unstated,stated,mixed}/summary_{dm,dsm,gm}.md`

---

## Output Directory Layout

```
output/
├── shared/
│   ├── activity_events.pkl     # condition-independent
│   ├── decisions.pkl           # identical across all conditions
│   └── c1_stated_goals.pkl     # Convictional stated goals (cache)
├── unstated/                   # LLM-mined unstated goals only
│   ├── step1_candidates.pkl
│   ├── step3_canonical_goals.pkl
│   ├── step5_final_goal_set.pkl
│   ├── mapping_{dm,dsm,gm}.pkl
│   ├── mapping_{dm,dsm,gm}.md  # schema-masked raw artifact
│   ├── summary_{dm,dsm,gm}.md  # obfuscation layer output (~525 words)
│   └── judge_{dm,dsm,gm}_T0.0.pkl
├── stated/                     # human-written stated goals only
│   └── ...
├── mixed/                      # both sources merged
│   └── ...
├── results_matrix.{csv,json}
└── calibration_pilot.pkl
```
