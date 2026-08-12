# LoRA Cassettes

Episodic LoRA adapter training for encoder-based retrieval. Train lightweight adapters (~10MB) that swap in/out like cassette tapes, fine-tuned on specific corpus slices for improved retrieval quality.

**Key Features:**
- 🎯 Retrieval-only (no generation)
- 🔄 Hot-swappable adapters (frozen base model + LoRA)
- 📈 Contrastive learning with in-batch negatives
- ⚡ MPS-accelerated training on Apple Silicon
- 💾 Episodic training with catastrophic forgetting prevention

See [`PLAN.md`](PLAN.md) for full implementation plan and architecture.

---

## Requirements

- **Python:** 3.13
- **Database:** PostgreSQL 15+ with pgvector extension
- **Hardware:** M-series Mac (MPS) recommended, or CUDA GPU
- **Memory:** 8GB+ RAM (128GB recommended for large batches)
- **Dependencies:** Managed via `uv`

---

## New User Setup

### 1. Install Dependencies

```bash
# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync
```

### 2. Database Setup

**Create the experiment database:**

```bash
createdb lora_cassettes -O $(whoami)
```

**Enable required extensions:**

```bash
psql lora_cassettes << 'EOF'
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
EOF
```

### 3. Populate Content Table

> **Data note.** This experiment was developed against a copy of a production database, and the
> original instructions here copied rows out of that backup. That data is not part of this
> repository and those instructions have been removed. **You need a synthetic data generator to
> run this** — see the TODO under "Populate content" below; it was never written.

Create the tables the pipeline expects:

```bash
# Create base tables
psql lora_cassettes << 'EOF'
CREATE TABLE IF NOT EXISTS organization (
  id uuid NOT NULL PRIMARY KEY,
  created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  name varchar(255),
  domain varchar(255)
);

CREATE TABLE IF NOT EXISTS content (
  id uuid NOT NULL PRIMARY KEY,
  created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  category varchar(255) NOT NULL,
  source varchar(255) NOT NULL,
  source_id text NOT NULL,
  title text NOT NULL,
  index_content text NOT NULL,
  author text,
  metadata jsonb NOT NULL,
  embedding vector(1536) NOT NULL DEFAULT '[0]'::vector,
  last_indexed_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
  organization_id uuid NOT NULL,
  content_type varchar(255) NOT NULL,
  source_url text NOT NULL,
  preview_content text,
  allowed_user_ids jsonb NOT NULL DEFAULT '[]'::jsonb,
  sharing varchar(255) NOT NULL DEFAULT 'private',
  tags jsonb NOT NULL DEFAULT '[]'::jsonb,
  CONSTRAINT content_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES organization(id) ON DELETE CASCADE
);

CREATE INDEX idx_content_sharing ON content(sharing);
CREATE INDEX idx_content_org_id ON content(organization_id);
CREATE INDEX idx_content_created_at ON content(created_at);
EOF
```

**Populate content — generate synthetic data**

This is the only supported path in the published version, and the generator does not exist yet:

```bash
# TODO: Create synthetic data generator script
# PYTHONPATH=. uv run python scripts/generate_synthetic_content.py --num-items 1000
```

Until that script is written, the training and evaluation steps below cannot be run end to end.

**Verify content loaded:**

```bash
psql lora_cassettes -c "SELECT COUNT(*), sharing FROM content GROUP BY sharing;"
```

Expected output: Thousands of rows with `sharing = 'organization'`

### 4. Apply Migrations

```bash
psql lora_cassettes -f migrations/001_create_training_schema.sql
```

Verify tables created:

```bash
psql lora_cassettes -c "\dt"
```

Expected tables: `episodes`, `training_pairs`, `adapters`, `eval_queries`, `eval_results`

### 5. Download Base Model

```bash
PYTHONPATH=. uv run python -c "
from src.model.download import EncoderModelDownloader
import asyncio

async def download():
    downloader = EncoderModelDownloader()
    model, tokenizer = await downloader.download_encoder_model()
    await downloader.save_model(model, tokenizer)
    print('✓ Base model downloaded: src/model/base_models/e5-base-v2/')

asyncio.run(download())
"
```

### 6. Create Episode 0

Mine training pairs from your content:

```bash
# Full dataset
PYTHONPATH=. uv run python scripts/create_episode_0.py --force

# Or start with a subset for testing
PYTHONPATH=. uv run python scripts/create_episode_0.py \
  --min-thread-size 3 \
  --max-pairs-per-thread 5 \
  --force
```

Verify pairs created:

```bash
psql lora_cassettes -c "SELECT COUNT(*) FROM training_pairs;"
```

---

## Quick Start

### Train Your First Adapter

**Validation run (100 pairs, fast test):**

```bash
PYTHONPATH=. TOKENIZERS_PARALLELISM=false uv run python scripts/train_episode_0.py \
  --subset 100 \
  --epochs 5
```

**Full training (5K pairs, production):**

```bash
PYTHONPATH=. TOKENIZERS_PARALLELISM=false uv run python scripts/train_episode_0.py
```

**Custom hyperparameters:**

```bash
PYTHONPATH=. TOKENIZERS_PARALLELISM=false uv run python scripts/train_episode_0.py \
  --batch-size 1024 \
  --epochs 10 \
  --lr 1e-4 \
  --checkpoint-every 5
```

### Test Your Adapter

```bash
PYTHONPATH=. uv run python << 'EOF'
from src.model.encoder import LoRAEncoderRetriever

# Load encoder with adapter
encoder = LoRAEncoderRetriever(
    base_model_path='src/model/base_models/e5-base-v2',
    device='mps'
)
encoder.load_adapter('src/model/adapters/episode_0/final')

# Encode some queries
queries = [
    "How do I implement authentication?",
    "What's the database schema for users?",
]
embeddings = encoder.encode(queries)
print(f"✓ Encoded {len(queries)} queries: {embeddings.shape}")
EOF
```

---

## CLI Reference

### Episode Management

**`scripts/create_episode_0.py`** - Create initial training dataset

```bash
PYTHONPATH=. uv run python scripts/create_episode_0.py [OPTIONS]

Options:
  --min-thread-size INT       Minimum items in thread to mine (default: 2)
  --max-pairs-per-thread INT  Max pairs per thread (default: unlimited)
  --force                     Use existing episode without prompting
  --dry-run                   Don't insert, just show what would be created
```

### Training

**`scripts/train_episode_0.py`** - Train LoRA adapter

```bash
PYTHONPATH=. TOKENIZERS_PARALLELISM=false uv run python scripts/train_episode_0.py [OPTIONS]

Data Options:
  --episode INT          Episode number to train on (default: 0)
  --subset INT           Use only first N pairs for testing (default: all)

Training Options:
  --epochs INT           Number of epochs (default: 20)
  --batch-size INT       Batch size (default: 512 for M3 Max)
  --lr FLOAT             Learning rate (default: 2e-4)

Hardware Options:
  --device STR           Device: mps|cuda|cpu (default: mps)

Checkpointing:
  --checkpoint-every INT Save every N epochs, 0=final only (default: 5)

Examples:
  # Quick validation
  ... train_episode_0.py --subset 100 --epochs 5

  # Full training with larger batches
  ... train_episode_0.py --batch-size 1024 --epochs 20

  # CPU training (slower)
  ... train_episode_0.py --device cpu
```

---

## Project Structure

```
lora_cassettes/
├── migrations/              # Database schema migrations
│   ├── 001_create_training_schema.sql
│   └── README.md
├── scripts/                 # Executable scripts
│   ├── create_episode_0.py  # Episode initialization
│   └── train_episode_0.py   # Training script
├── src/
│   ├── data/                # Data layer (Pydantic + asyncpg)
│   │   ├── models.py        # Type-safe data models
│   │   └── db.py            # Database helpers
│   ├── mining/              # Pair mining strategies
│   │   └── github.py        # GitHub thread mining
│   ├── model/               # Model infrastructure
│   │   ├── download.py      # Model downloader
│   │   ├── encoder.py       # LoRA encoder wrapper
│   │   └── base_models/     # Downloaded models
│   │       └── e5-base-v2/  # Base encoder (419MB)
│   └── training/            # Training infrastructure
│       ├── config.py        # Training configuration
│       ├── dataset.py       # PyTorch Dataset
│       └── contrastive.py   # InfoNCE loss + trainer
├── PLAN.md                  # Full implementation plan
├── README.md                # This file
└── pyproject.toml           # Dependencies
```

---

## Development Workflow

### 1. Create New Episode

When you have new content or want to refresh training:

```bash
# Create Episode 1 (with replay buffer from Episode 0)
PYTHONPATH=. uv run python scripts/create_episode_1.py
```

### 2. Train Adapter

```bash
PYTHONPATH=. TOKENIZERS_PARALLELISM=false uv run python scripts/train_episode_1.py
```

### 3. Evaluate

```bash
# TODO: Evaluation pipeline
PYTHONPATH=. uv run python scripts/evaluate_adapter.py \
  --adapter-id "convx/episode_1-v1.0.0" \
  --query-set hee
```

### 4. Compare Adapters

```bash
# TODO: Comparison tool
psql lora_cassettes -c "
  SELECT adapter_id, metrics->>'recall@10' as recall_10, status
  FROM adapters
  ORDER BY created_at DESC
  LIMIT 5;
"
```

---

## Troubleshooting

### Out of Memory

Reduce batch size:

```bash
... train_episode_0.py --batch-size 256
```

### Slow Training

Check device usage:

```bash
# Should see MPS device in use
... train_episode_0.py --subset 10 --epochs 1
# Look for "Device: mps" in output
```

### Database Connection Issues

Verify database exists and is accessible:

```bash
psql lora_cassettes -c "SELECT COUNT(*) FROM content;"
```

### TOKENIZERS_PARALLELISM Warning

Set environment variable:

```bash
export TOKENIZERS_PARALLELISM=false
# Or prefix each command
TOKENIZERS_PARALLELISM=false uv run python ...
```

---

## Performance Tips

### M3 Max Optimization

- **Batch size:** 512-1024 (leverage 128GB RAM)
- **Workers:** 8 (50% of CPU cores)
- **Device:** `mps` (use Apple GPU)
- **Mixed precision:** Enabled by default (2x speedup)

### Expected Training Times

| Dataset Size | Epochs | Batch Size | Device | Time     |
|--------------|--------|------------|--------|----------|
| 100 pairs    | 5      | 512        | MPS    | ~10 min  |
| 1K pairs     | 20     | 512        | MPS    | ~20 min  |
| 5K pairs     | 20     | 512        | MPS    | ~30 min  |
| 50K pairs    | 20     | 1024       | MPS    | ~3 hours |

---

## Architecture Notes

### Why LoRA?

- **Small:** 10MB vs 400MB full model
- **Fast:** Train in minutes vs hours
- **Safe:** Base model stays frozen
- **Stackable:** Multiple adapters for different domains

### Why Contrastive Learning?

- **Unsupervised:** No manual labels needed
- **Effective:** Learn from implicit relationships (thread context)
- **Scalable:** In-batch negatives (512 pairs → 511 negatives each)

### Why Episodic?

- **Freshness:** Incorporate new content regularly
- **Stability:** Replay buffer prevents catastrophic forgetting
- **Versioned:** Track performance over time

---

## Contributing

See [`PLAN.md`](PLAN.md) for implementation roadmap.

**Current Status:**
- ✅ Database schema and migrations
- ✅ Base model infrastructure
- ✅ GitHub pair mining
- ✅ Episode 0 creation
- ✅ Contrastive training pipeline
- ⏳ Retrieval pipeline (next)
- ⏳ Evaluation harness (next)
- ⏳ Router heuristics (next)

---

## License

Internal Convictional research project.
