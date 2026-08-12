-- Migration 001: Create training data schema for LoRA Cassettes experiment
-- Based on PLAN.md sections 6-7 (Episodic Fine-Tuning, Adapter Registry)
-- and sections 9-11 (Harness Integration, Experiment Matrix, Metrics)
--
-- This schema supports:
-- - Episodic training with replay buffers
-- - Adapter versioning and registry
-- - Training pair storage and management
-- - Evaluation query management (HEE set + stability set)
-- - Retrieval evaluation results tracking

-- =============================================================================
-- EPISODES: Track training episodes over time
-- =============================================================================
CREATE TABLE IF NOT EXISTS episodes (
  id SERIAL PRIMARY KEY,
  episode_num INT NOT NULL UNIQUE,
  start_date TIMESTAMPTZ NOT NULL,
  end_date TIMESTAMPTZ NOT NULL,
  corpus_snapshot_date TIMESTAMPTZ NOT NULL,
  num_new_chunks INT NOT NULL DEFAULT 0,
  num_updated_chunks INT NOT NULL DEFAULT 0,
  status VARCHAR(50) NOT NULL DEFAULT 'pending',
  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  completed_at TIMESTAMPTZ,

  CONSTRAINT episodes_status_check CHECK (status IN ('pending', 'training', 'completed', 'failed'))
);

COMMENT ON TABLE episodes IS 'Tracks training episodes for incremental adapter updates';
COMMENT ON COLUMN episodes.episode_num IS 'Sequential episode number (0 = initial, 1+ = incremental)';
COMMENT ON COLUMN episodes.corpus_snapshot_date IS 'Date of corpus snapshot used for training';
COMMENT ON COLUMN episodes.num_new_chunks IS 'Number of new content chunks in this episode';
COMMENT ON COLUMN episodes.num_updated_chunks IS 'Number of updated content chunks in this episode';

-- =============================================================================
-- TRAINING_PAIRS: Unsupervised positive/negative pairs for contrastive learning
-- =============================================================================
CREATE TABLE IF NOT EXISTS training_pairs (
  id BIGSERIAL PRIMARY KEY,
  episode_id INT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
  anchor_content_id UUID NOT NULL REFERENCES content(id) ON DELETE CASCADE,
  positive_content_id UUID NOT NULL REFERENCES content(id) ON DELETE CASCADE,
  negative_content_id UUID REFERENCES content(id) ON DELETE CASCADE,

  -- Pair mining metadata (from PLAN.md section 5)
  pair_type VARCHAR(50) NOT NULL, -- e.g., 'adjacent_section', 'same_thread', 'consecutive_reply'
  mining_method VARCHAR(50) NOT NULL, -- 'heuristic', 'bm25_hard_neg', 'ance_mining'
  source_family VARCHAR(50) NOT NULL, -- 'github', 'docs', 'email', 'meetings', 'code'

  -- Cached text to avoid joins during training
  anchor_text TEXT NOT NULL,
  positive_text TEXT NOT NULL,
  negative_text TEXT,

  -- Replay buffer flag for catastrophic forgetting prevention (PLAN.md section 6)
  is_in_replay_buffer BOOLEAN NOT NULL DEFAULT FALSE,

  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

  CONSTRAINT training_pairs_pair_type_check CHECK (pair_type IN (
    'adjacent_section', 'same_heading', 'same_thread', 'consecutive_reply',
    'qa_turn', 'parent_reply', 'message_attachment', 'symbol_docstring',
    'header_implementation', 'callsite_definition'
  )),
  CONSTRAINT training_pairs_mining_method_check CHECK (mining_method IN (
    'heuristic', 'bm25_hard_neg', 'ance_mining', 'in_batch'
  )),
  CONSTRAINT training_pairs_source_check CHECK (source_family IN (
    'github', 'docs', 'email', 'meetings', 'code', 'mixed'
  ))
);

CREATE INDEX idx_training_pairs_episode ON training_pairs(episode_id);
CREATE INDEX idx_training_pairs_replay ON training_pairs(is_in_replay_buffer) WHERE is_in_replay_buffer = TRUE;
CREATE INDEX idx_training_pairs_source ON training_pairs(source_family);
CREATE INDEX idx_training_pairs_anchor ON training_pairs(anchor_content_id);

-- Partial unique index for pairs with explicit negatives
CREATE UNIQUE INDEX idx_training_pairs_unique_with_neg
  ON training_pairs(episode_id, anchor_content_id, positive_content_id, negative_content_id)
  WHERE negative_content_id IS NOT NULL;

COMMENT ON TABLE training_pairs IS 'Stores anchor-positive-negative triplets for contrastive learning';
COMMENT ON COLUMN training_pairs.pair_type IS 'Type of relationship between anchor and positive (see PLAN.md section 5)';
COMMENT ON COLUMN training_pairs.mining_method IS 'Method used to mine this pair (heuristic, hard negative, etc.)';
COMMENT ON COLUMN training_pairs.is_in_replay_buffer IS 'Part of 1-2% replay set to prevent catastrophic forgetting';

-- =============================================================================
-- ADAPTERS: Registry of LoRA adapters with metadata and metrics
-- =============================================================================
CREATE TABLE IF NOT EXISTS adapters (
  id SERIAL PRIMARY KEY,
  adapter_id VARCHAR(255) NOT NULL UNIQUE, -- Semantic versioning: 'convx/support-2025q4-v1.0.0'
  base_model VARCHAR(255) NOT NULL, -- e.g., 'e5-base-v2', 'gte-base'
  episode_id INT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,

  -- Training configuration (PLAN.md sections 6-7)
  sources JSONB NOT NULL, -- Array: ['docs', 'slack', 'email', 'meetings', 'code']
  objective VARCHAR(50) NOT NULL, -- 'contrastive', 'retromae'
  train_start_date TIMESTAMPTZ NOT NULL,
  train_end_date TIMESTAMPTZ NOT NULL,
  replay_pct FLOAT NOT NULL,

  -- Index mapping (PLAN.md section 8.2)
  hnsw_index_id VARCHAR(255),

  -- Hyperparameters
  lora_config JSONB NOT NULL, -- {r: 16, alpha: 32, dropout: 0.05, target_modules: [...]}
  training_config JSONB NOT NULL, -- {lr: 2e-4, batch_size: 256, steps: 20000, warmup: 0.05}

  -- Evaluation metrics (PLAN.md section 11)
  metrics JSONB NOT NULL, -- {recall@5, recall@10, ndcg@10, ndcg@20, freshness_recall}
  stability_delta FLOAT, -- Change in recall@10 on stability set vs Episode 0

  -- Lifecycle management
  status VARCHAR(50) NOT NULL DEFAULT 'training', -- 'training', 'active', 'deprecated'
  storage_path TEXT NOT NULL, -- Path to .safetensors file
  created_by VARCHAR(255) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  promoted_at TIMESTAMPTZ, -- When promoted to 'active' status

  CONSTRAINT adapters_objective_check CHECK (objective IN ('contrastive', 'retromae')),
  CONSTRAINT adapters_status_check CHECK (status IN ('training', 'active', 'deprecated', 'failed')),
  CONSTRAINT adapters_replay_pct_check CHECK (replay_pct >= 0 AND replay_pct <= 1)
);

CREATE INDEX idx_adapters_status ON adapters(status);
CREATE INDEX idx_adapters_episode ON adapters(episode_id);
CREATE INDEX idx_adapters_created_at ON adapters(created_at DESC);

COMMENT ON TABLE adapters IS 'Registry of LoRA adapters with versioning and metrics (PLAN.md section 7)';
COMMENT ON COLUMN adapters.adapter_id IS 'Semantic version: convx/<domain>-<YYYYqN>-v<M.m.p>';
COMMENT ON COLUMN adapters.stability_delta IS 'Change in Recall@10 on stability set; promotion gate checks >-1%';
COMMENT ON COLUMN adapters.storage_path IS 'Path to LoRA delta weights (.safetensors, 20-80 MB)';

-- =============================================================================
-- EVAL_QUERIES: Curated evaluation queries (HEE set + stability set)
-- =============================================================================
CREATE TABLE IF NOT EXISTS eval_queries (
  id SERIAL PRIMARY KEY,
  query_text TEXT NOT NULL,

  -- Query characteristics (PLAN.md section 4)
  query_type VARCHAR(50) NOT NULL, -- 'alias_heavy', 'cross_source', 'cross_team', 'temporal'
  difficulty VARCHAR(20) NOT NULL, -- 'easy', 'medium', 'hard'

  -- Expected results metadata
  expected_sources JSONB, -- Array: which sources should have relevant results
  ground_truth_content_ids JSONB, -- Array of UUIDs for relevant content

  -- Organization
  tags JSONB NOT NULL DEFAULT '[]'::jsonb,
  is_in_stability_set BOOLEAN NOT NULL DEFAULT FALSE, -- ~2k chunks, never retrained on

  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

  CONSTRAINT eval_queries_type_check CHECK (query_type IN (
    'alias_heavy', 'cross_source', 'cross_team', 'temporal', 'general'
  )),
  CONSTRAINT eval_queries_difficulty_check CHECK (difficulty IN ('easy', 'medium', 'hard'))
);

CREATE INDEX idx_eval_queries_stability ON eval_queries(is_in_stability_set) WHERE is_in_stability_set = TRUE;
CREATE INDEX idx_eval_queries_type ON eval_queries(query_type);
CREATE INDEX idx_eval_queries_difficulty ON eval_queries(difficulty);

COMMENT ON TABLE eval_queries IS 'HEE (high-entropy entity) query set and stability set for evaluation';
COMMENT ON COLUMN eval_queries.query_type IS 'Type of hard query (alias-heavy, cross-source, etc.)';
COMMENT ON COLUMN eval_queries.is_in_stability_set IS 'Part of stability set (~2k chunks, never retrained on)';

-- =============================================================================
-- EVAL_RESULTS: Retrieval evaluation results per adapter per query
-- =============================================================================
CREATE TABLE IF NOT EXISTS eval_results (
  id BIGSERIAL PRIMARY KEY,
  adapter_id INT REFERENCES adapters(id) ON DELETE CASCADE,
  eval_query_id INT NOT NULL REFERENCES eval_queries(id) ON DELETE CASCADE,

  -- Retrieval results
  retrieved_content_ids JSONB NOT NULL, -- Ordered array of content UUIDs
  scores JSONB NOT NULL, -- Ordered array of scores matching content_ids
  adapters_used JSONB, -- Array of adapter_ids selected by router (PLAN.md section 8.1)

  -- Computed metrics (PLAN.md section 11)
  recall_at_5 FLOAT,
  recall_at_10 FLOAT,
  recall_at_20 FLOAT,
  ndcg_at_10 FLOAT,
  ndcg_at_20 FLOAT,

  -- Performance metrics
  latency_ms INT,

  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

  CONSTRAINT eval_results_unique_adapter_query UNIQUE(adapter_id, eval_query_id, created_at)
);

CREATE INDEX idx_eval_results_adapter ON eval_results(adapter_id);
CREATE INDEX idx_eval_results_query ON eval_results(eval_query_id);
CREATE INDEX idx_eval_results_created_at ON eval_results(created_at DESC);

COMMENT ON TABLE eval_results IS 'Retrieval evaluation results tracking metrics per adapter per query';
COMMENT ON COLUMN eval_results.adapters_used IS 'Which adapters were selected by router for this query';
COMMENT ON COLUMN eval_results.latency_ms IS 'End-to-end retrieval latency (target P95 <= 900ms)';
