# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Training & Experiments
```bash
# Main training with real authorship data (local development)
poetry run python scripts/train.py

# Export training data to GCS dataset (for Vertex AI)
poetry run python scripts/export_vertex_dataset.py

# Submit training job to Vertex AI (working command)
poetry run python scripts/submit_vertex_job.py \
  --project-id ${GCP_PROJECT} \
  --image-uri gcr.io/${GCP_PROJECT}/writing-identification:v38 \
  --gcs-bucket ${GCS_BUCKET} \
  --experiment-name baseline_run \
  --machine-type n1-standard-8 \
  --accelerator-type NVIDIA_TESLA_T4 \
  --display-name "authorship-training-$(date +%Y%m%d-%H%M%S)"

# Submit predefined experimental configurations
poetry run python scripts/submit_vertex_job.py \
  --project-id ${GCP_PROJECT} \
  --image-uri gcr.io/${GCP_PROJECT}/writing-identification:v38 \
  --gcs-bucket ${GCS_BUCKET} \
  --experiment-config external_contrastive \
  --display-name "authorship-reddit-$(date +%Y%m%d-%H%M%S)"

# Submit all experiments in batch
./scripts/submit_experiments_batch.sh

# Extract and analyze training data from database (local development)
poetry run python -m data.extract_training_data
```

### Pre-computed Features (Performance Optimization)
```bash
# Pre-compute features for massive speedup (1200x+ faster training)
# This extracts all features once and saves to HDF5 files with separated semantic/style features

# FAST MODE - Fit and save extractors only (minutes vs 2.5 days)
poetry run python scripts/precompute_features.py --config external_contrastive --extractors-only

# FULL MODE - Pre-compute features for external dataset configurations (takes ~2.5 days)
poetry run python scripts/precompute_features.py --config external_contrastive --upload-to-gcs
poetry run python scripts/precompute_features.py --config external_triplet --upload-to-gcs
poetry run python scripts/precompute_features.py --config external_attention --upload-to-gcs

# Optional parameters
poetry run python scripts/precompute_features.py --config external_contrastive --batch-size 500 --upload-to-gcs
poetry run python scripts/precompute_features.py --config external_contrastive --resume --upload-to-gcs
poetry run python scripts/precompute_features.py --config external_contrastive --max-samples 10000  # For testing

# Test pre-computed features integrity
poetry run python scripts/test_precomputed_features.py

# Configurations with pre-computed features automatically use them when available:
# - Features stored separately: semantic (768D) + style+email (1169D + 38D)
# - Fitted extractors saved as .extractors.pkl files alongside HDF5 features
# - Compatible with all encoder types: fusion, attention, simple
# - Memory-mapped HDF5 loading for efficient training
# - Supports all external dataset configurations
```

### Baseline Evaluation
```bash
# Run all baseline models (LUAR, ModernBERT, Claude Haiku 3.5, Sonnet 4, Opus 4.1)
poetry run python scripts/run_baseline_evaluation.py

# Run only LLM baselines (with automatic text sanitization to prevent data leakage)
poetry run python scripts/run_baseline_evaluation.py --models llm-haiku llm-sonnet llm-opus

# Run only traditional baselines
poetry run python scripts/run_baseline_evaluation.py --models luar modernbert

# Run custom pre-trained Siamese model baseline
poetry run python scripts/run_baseline_evaluation.py --models custom --checkpoint-path models/checkpoints/best_model.pt

# Run specific model with custom rate limiting and no text sanitization
poetry run python scripts/run_baseline_evaluation.py --models llm-haiku --max-concurrent 5 --rate-limit 50 --no-sanitize

# Test LLM baseline with small sample
poetry run python scripts/test_llm_baseline.py
```

### Testing & Analysis
```bash
# Test complete feature extraction pipeline
poetry run python -m scripts.test_data_pipeline

# Analyze content types in dataset
poetry run python -m scripts.analyze_content_types

# Check author pair statistics for training
poetry run python -m scripts.analyze_pairs

# Check all available content types in database
poetry run python -m scripts.check_all_content_types

# Analyze GitHub authors in database
poetry run python -m scripts.check_github_authors

# Test GCS data loader functionality
poetry run python scripts/test_gcs_loader.py
```

## Architecture

This is a **Siamese neural network** system for authorship verification that determines if two text samples were written by the same person.

### Core Pipeline
1. **Feature Extraction** (`features/`) - Extracts three types of features from text
2. **Siamese Network** (`models/siamese.py`) - Twin networks that compare feature embeddings
3. **Training System** (`scripts/train.py`) - Handles model training with early stopping
4. **Data Loading** (`data/`) - Manages training data from database or GCS

### Key Architectural Concepts

**Multi-Modal Feature Fusion** (`features/extractors.py`)
- **Semantic Features**: Sentence-BERT embeddings using `all-mpnet-base-v2` (768D)
- **Stylometric Features**: Character/word n-grams, POS tags, function words (1169D actual)
- **Email-Specific Features**: Greeting/closing patterns, reply structure (38D)
- **Fitted Extractor Persistence**: Critical fix - extractors now saved with models for inference
- Features are cached after first extraction for faster subsequent runs

**Siamese Network Architecture** (`models/siamese.py`)
- Twin networks with shared weights process text pairs
- Multiple encoder options: `FeatureFusionEncoder`, `SimpleEncoder`, `AttentionEncoder`
- Uses cosine similarity loss with margin-based contrastive learning
- Output: similarity score for same/different author classification

**Training Strategy** (`data/dataset.py`)
- Hard-negative mining: selects top-k hardest negative pairs per batch
- Balanced positive/negative pair generation
- Dynamic threshold optimization using validation data
- Filters authors with minimum sample count (default: 5)

**Dual Environment Support**
- **Local Development**: Uses PostgreSQL database + MPS acceleration (Apple Silicon)
- **Vertex AI**: Uses GCS datasets + CUDA GPUs for distributed training

## Configuration

Configuration is managed through Pydantic models in `config/config.py`:

### Key Settings
- **Model Architecture**: 768D semantic embeddings + 1207D stylometric+email features → 256D final embeddings
- **Training**: Batch size 64, learning rate 1e-5, margin 0.2, early stopping patience 10
- **Data**: Min text length 100 chars, validation split 20%, minimum 5 samples per author
- **Device**: Auto-detects MPS (Apple Silicon) or CUDA (Vertex AI) or falls back to CPU
- **Extractors**: Fitted TF-IDF vectorizers saved with checkpoints for inference

### Environment Variables
- `DATABASE_URL`: PostgreSQL connection (local development)
- `GCP_PROJECT`: Google Cloud project ID (Vertex AI)
- `GCS_BUCKET`: Storage bucket for datasets and models (Vertex AI)

## Directory Structure
```
writing_identification/
├── config/                 # Configuration (Pydantic models)
├── data/                   # Data loading (database + GCS)
├── features/               # Multi-modal feature extraction
├── models/                 # Siamese network + encoders + losses
├── baselines/              # Baseline models (LUAR, ModernBERT, LLMs, Custom)
├── scripts/                # Training scripts + utilities
├── utils/                  # GCS storage helpers
└── results/                # Training outputs + checkpoints
```

## Key Implementation Details

1. **Pre-computed Features**: HDF5-based feature pre-computation with separated semantic/style storage for 1200x+ training speedup
2. **Feature Architecture**: Semantic features (768D) and style+email features (1207D) stored separately for encoder compatibility
3. **Hard-Negative Mining**: Training uses top-k hardest negatives per batch for better discrimination
4. **Dynamic Thresholding**: Uses median similarity score as classification threshold during validation
5. **MPS Compatibility**: Multiprocessing disabled for Apple Silicon compatibility
6. **Dual Data Sources**: Local PostgreSQL database OR GCS dataset (Vertex AI)
7. **Multi-Encoder Support**: Fusion, attention, and simple encoders work with both on-the-fly and pre-computed features
8. **Extractor Persistence**: Fitted TF-IDF vectorizers saved with checkpoints - critical for inference
9. **LLM Baseline Integration**: Claude models with async rate limiting and text sanitization
10. **Custom Model Baseline**: Load and evaluate pre-trained Siamese models from checkpoints
11. **Text Sanitization**: Prevents LLM data leakage by removing author metadata from evaluation text

## Vertex AI Setup

### Service Account & Authentication
- **Account**: `vertex-training-sa@${GCP_PROJECT}.iam.gserviceaccount.com`
- **Roles**: Storage Admin, Vertex AI User
- **Docker Image**: `gcr.io/${GCP_PROJECT}/writing-identification:v38` (latest with two-stage training, Reddit data, and extractor persistence)

### Data Sources
- **Local Development**: PostgreSQL `decide_development` database
- **Vertex AI Training**: GCS dataset `gs://${GCS_BUCKET}/datasets/authorship_dataset.jsonl`
- **Requirements**: Minimum 5 samples per author, 100+ characters per sample

### Docker Build Process
```bash
# Build and push new version
docker build -t gcr.io/${GCP_PROJECT}/writing-identification:v<NEW> .
docker push gcr.io/${GCP_PROJECT}/writing-identification:v<NEW>

# Or use automated script
./scripts/build_and_deploy.sh --project-id ${GCP_PROJECT} --tag v<NEW>
```

## Performance Notes

- **Training Time**: ~2-3 hours for 50 epochs on MPS/T4 GPU
- **Feature Extraction**:
  - On-the-fly: ~5-10 minutes per epoch (bottleneck)
  - Pre-computed: ~0.5 seconds per epoch (1200x+ speedup)
- **Pre-computation Time**:
  - Full pipeline: ~2.5 days for large datasets
  - Extractors-only: ~5-10 minutes (use --extractors-only flag)
- **Inference**: <100ms per text pair comparison (requires fitted extractors)
- **Memory**: ~2GB GPU memory for batch_size=64 (pre-computed features reduce memory usage)
- **Text Requirements**: 100+ characters minimum, 1000+ words optimal
- **Storage**:
  - Pre-computed features: ~500MB-2GB per dataset (HDF5 format with compression)
  - Fitted extractors: ~400MB per dataset (.extractors.pkl files)
- **LLM Baseline Performance**: Async rate limiting with 10 concurrent requests, 100 requests/minute
