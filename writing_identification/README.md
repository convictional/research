# Authorship Verification System

A Siamese neural network system for verifying authorship using semantic embeddings and stylometric analysis, trained on internal team data and Reddit discussion data.

> ### Read this first
>
> **The headline result is close to chance.** The best AUC from a Claude baseline here is **0.553**
> — an AUC of 0.5 is a coin flip. The accuracy figures below look more encouraging than they are:
> on a balanced pairs task with a tuned threshold, 63–73% accuracy alongside an AUC of ~0.55 means
> the models are barely separating same-author from different-author pairs. **Nothing here is
> deployable as an authenticity signal, and it should not be read as evidence that authorship
> verification works on this kind of text.** The custom model and the two traditional baselines
> reach higher AUC (0.66–0.69) but at precision/recall trade-offs that are worse in practice.
>
> **The training data is not included, by design.** The internal portion of the corpus was
> contributed by colleagues on an opt-in basis for this experiment only. Publishing it would break
> that basis, so it was removed before open-sourcing, along with the author roster the scripts
> referenced. The code ships with placeholder author names you must replace with your own. The
> Reddit portion is publicly available but is likewise not redistributed here.
>
> **On the ethics of building this at all**, including why the intended product framing was
> sender-opt-in rather than recipient-side detection, and what was deliberately ruled out, see
> [EXPERIMENT_OVERVIEW.md](EXPERIMENT_OVERVIEW.md). Please read that before reusing any of this.

## Performance Results

Benchmark results on internal validation data (630 test pairs, September 2025). Note the AUC
column, not the accuracy column — see the caveat above:

| Model | Accuracy | AUC | Precision | Recall | F1-Score |
|-------|----------|-----|-----------|--------|----------|
| **Claude Opus** | 72.86% | 55.30% | 78.35% | 63.17% | 69.95% |
| **Claude Sonnet** | 67.14% | 55.36% | 68.37% | 63.81% | 66.01% |
| **Custom Model** | 62.70% | 66.37% | 59.71% | 78.10% | 67.68% |
| **Claude Haiku** | 62.38% | 59.85% | 59.95% | 74.60% | 66.48% |
| **ModernBERT** | 58.41% | 67.85% | 74.77% | 25.40% | 37.91% |
| **LUAR** | 57.94% | 68.69% | 59.84% | 48.25% | 53.43% |

Claude models demonstrate superior accuracy with balanced precision/recall trade-offs. ModernBERT and LUAR achieve higher AUC scores but with conservative precision-favored predictions.

## Architecture

**Siamese neural networks** with multi-modal feature fusion:
- **Semantic features**: 768D sentence embeddings (all-mpnet-base-v2)
- **Stylometric features**: 1,169D character/word patterns, POS tags, function words
- **Email patterns**: Greeting/closing detection, reply structure analysis

**Training approaches**:
- Two-stage training: classification pretraining → verification fine-tuning
- Reddit discussion data for robust generalization
- Internal team data for domain-specific validation
- Pre-computed features for 1200x+ training speedup

## Directory Structure

```
writing_identification/
├── README.md                   # This file
├── config/
│   ├── config.py              # Base configuration
│   └── experiment_configs.py   # Predefined experiment configurations
├── scripts/
│   ├── train.py               # Main training script
│   ├── vertex_train.py        # Vertex AI training
│   ├── precompute_features.py # Pre-compute features for speedup
│   ├── test_precomputed_features.py  # Test pre-computed features
│   ├── submit_vertex_job.py   # Job submission
│   ├── run_baseline_evaluation.py  # Full baseline evaluation
│   └── test_baselines_local.py     # Quick baseline testing
│
├── data/                       # Data handling
│   ├── extract_training_data.py  # Internal database extraction
│   ├── external_datasets.py      # Reddit dataset loading
│   ├── reddit_pairs.py           # Reddit pair generation for verification
│   ├── precomputed_dataset.py    # HDF5 pre-computed features dataset
│   └── dataset.py               # PyTorch datasets
│
├── features/                   # Feature extraction
│   ├── extractors.py           # Semantic & stylometric features
│   └── email_patterns.py       # Email-specific patterns
│
├── models/                     # Neural network architectures
│   ├── siamese.py              # Siamese network implementation
│   ├── encoder.py              # Multiple encoder options
│   └── losses.py               # Contrastive/triplet loss functions
│
├── baselines/                  # Baseline model evaluation (local only)
│   ├── base.py                # Abstract baseline interface
│   ├── luar.py               # LUAR baseline implementation
│   ├── modernbert.py         # ModernBERT baseline
│   ├── llm.py                # Claude LLM baselines (Haiku, Sonnet, Opus)
│   ├── custom_model.py       # Custom pre-trained Siamese model baseline
│   ├── text_sanitizer.py     # Text sanitization for data leakage prevention
│   ├── evaluate.py           # Evaluation pipeline
│   └── results/              # Baseline evaluation outputs
│
├── utils/                      # Utilities
│   └── gcs.py                 # Google Cloud Storage helpers
│
└── results/                    # Training outputs (auto-created)
    ├── models/                 # Saved model checkpoints
    ├── training_history/       # Training metrics & logs
    └── cache/                 # Feature extraction cache
```

## CLI Commands

### Local Training

#### Internal Data Only (Legacy)
```bash
poetry run python scripts/train.py
```

#### Reddit Data Training
```bash
# Reddit-only training with contrastive loss
poetry run python scripts/train.py --config external_contrastive

# Reddit-only with triplet loss
poetry run python scripts/train.py --config external_triplet

# Reddit-only with attention encoder
poetry run python scripts/train.py --config external_attention

# Reddit pretraining + internal fine-tuning
poetry run python scripts/train.py --config external_pretrain

# Mixed Reddit + internal training
poetry run python scripts/train.py --config mixed
```

### Vertex AI Training

Submit training jobs to Google Cloud with GPU acceleration:

```bash
# Submit Reddit contrastive training
poetry run python scripts/submit_vertex_job.py \
  --project-id ${GCP_PROJECT} \
  --image-uri gcr.io/${GCP_PROJECT}/writing-identification:v38 \
  --gcs-bucket ${GCS_BUCKET} \
  --experiment-config external_contrastive \
  --machine-type n1-standard-8 \
  --accelerator-type NVIDIA_TESLA_T4 \
  --display-name "reddit-contrastive-$(date +%Y%m%d-%H%M%S)"

# Submit all experiments in batch
./scripts/submit_experiments_batch.sh
```

### Baseline Model Evaluation

Compare your custom model against established baselines:

```bash
# Run comprehensive baseline evaluation (all models)
poetry run python scripts/run_baseline_evaluation.py

# Run only LLM baselines (Claude models with text sanitization)
poetry run python scripts/run_baseline_evaluation.py --models llm-haiku llm-sonnet llm-opus

# Run only traditional baselines
poetry run python scripts/run_baseline_evaluation.py --models luar modernbert

# Evaluate custom pre-trained Siamese model
poetry run python scripts/run_baseline_evaluation.py --models custom --checkpoint-path models/checkpoints/best_model.pt

# Custom rate limiting for LLM models
poetry run python scripts/run_baseline_evaluation.py --models llm-haiku --max-concurrent 5 --rate-limit 50

# Disable text sanitization (for debugging - may cause data leakage)
poetry run python scripts/run_baseline_evaluation.py --models llm-haiku --no-sanitize
```

This evaluates LUAR, ModernBERT, Claude LLM models, and custom Siamese models:
- Performance metrics (accuracy, AUC, precision, recall, F1)
- Detailed confusion matrices with sensitivity/specificity
- CSV output with predictions, reasoning, and confidence scores
- JSON results for programmatic analysis
- Automatic text sanitization to prevent LLM data leakage

### Pre-computed Features (Performance Optimization)

Achieve 1200x+ training speedup by pre-computing features once and storing them in efficient HDF5 format. This happens automatically during training if no cache is available:

```bash
# FAST MODE - Fit and save extractors only (minutes vs 2.5 days)
poetry run python scripts/precompute_features.py --config external_contrastive --extractors-only

# FULL MODE - Pre-compute features for Reddit dataset configurations (takes ~2.5 days)
poetry run python scripts/precompute_features.py --config external_contrastive --upload-to-gcs
poetry run python scripts/precompute_features.py --config external_triplet --upload-to-gcs
poetry run python scripts/precompute_features.py --config external_attention --upload-to-gcs

# Optional parameters
poetry run python scripts/precompute_features.py --config external_contrastive --batch-size 500 --upload-to-gcs
poetry run python scripts/precompute_features.py --config external_contrastive --resume --upload-to-gcs
poetry run python scripts/precompute_features.py --config external_contrastive --max-samples 10000  # For testing

# Test pre-computed features integrity
poetry run python scripts/test_precomputed_features.py
```

### Data Analysis

```bash
# Analyze internal training data
poetry run python data.extract_training_data.py
```

## Training Data

### Internal Data (Domain-Specific)
- **Authors**: 8 team members with substantial writing samples
- **Content Types**: GitHub comments, issues, discussions, Google Docs, meetings
- **Total Samples**: 3,000+ filtered samples (500-10,000 characters each)
- **Usage**: Validation and domain-specific fine-tuning

### Reddit Data (Generalization)
- **Dataset**: Reddit discussions from business/tech/professional subreddits
- **Subreddits**: 15 professional communities (business, startups, sysadmin, finance, etc.)
- **Scale**: 5,000 authors with 10+ comments each
- **Filtering**: 100-5000 characters, excluding deleted/bot content
- **Caching**: Deterministic pair generation with lightweight manifests
- **Usage**: Primary training for robust generalization

### Training Strategies

1. **Two-Stage Training**: Classification on Reddit → Verification fine-tuning with pairs
2. **Reddit Only**: Train exclusively on Reddit data, validate on internal
3. **Reddit Pretrain**: Pre-train on Reddit, then fine-tune on internal
4. **Mixed Training**: Combined training on both Reddit and internal data

## Feature Engineering

### Semantic Features (768D)
- **Model**: Sentence-BERT `all-mpnet-base-v2`
- **Purpose**: Capture semantic meaning and context
- **Optimization**: MPS-accelerated inference on Apple Silicon

### Stylometric Features (1,169D)
- Character n-grams (2-4): Writing style patterns
- Word n-grams (1-2): Vocabulary preferences
- POS tag distributions: Grammatical patterns
- Function word frequencies: Structural preferences
- Punctuation patterns: Formatting habits
- Statistical metrics: Sentence length, complexity

### Email-Specific Features (38D)
- Greeting patterns: "Hi", "Hello", "Hey" variations
- Closing signatures: "Best", "Thanks", "Cheers"
- Reply formatting: Quoting styles, indentation
- Structure analysis: Paragraph breaks, formatting

## Model Architecture

### Siamese Network Design
- **Twin Networks**: Shared-weight architecture for similarity learning
- **Encoders**: Multiple options (fusion, attention, simple)
- **Output**: 256D final embeddings for similarity computation
- **Loss Functions**: Contrastive loss with margin or triplet loss with hard negatives

### Two-Stage Training
1. **Classification Stage**: Train on Reddit authors to learn representations
   - Uses cross-entropy loss with author labels
   - Learns discriminative features across many authors
   - Typically 10-20 epochs

2. **Verification Stage**: Fine-tune for pairwise similarity
   - Converts to Siamese architecture
   - Uses contrastive loss on author pairs
   - Transfers encoder weights from classification
   - Typically 10-15 epochs

### Encoder Options

1. **FeatureFusionEncoder**: Combines semantic + stylometric features
2. **AttentionEncoder**: Attention-based fusion for complex patterns
3. **SimpleEncoder**: Concatenation-based approach for baseline

### GPU Optimization
- **Batch Sizes**: 256 (contrastive), 128 (triplet/attention) - 4x larger than original
- **Memory Usage**: <1% of 16GB GPU memory (8.9MB utilization)
- **Acceleration**: MPS (Apple Silicon) or CUDA (Vertex AI)

## Configuration

### Available Experiment Configurations

| Config | Strategy | Loss | Batch Size | Features |
|--------|----------|------|------------|----------|
| `external_contrastive` | Reddit Only | Contrastive | 256 | All optimized training |
| `external_triplet` | Reddit Only | Triplet | 128 | Hard negative mining |
| `external_attention` | Reddit Only | Contrastive | 128 | Attention encoder |
| `external_pretrain` | Pretrain + Finetune | Contrastive | 64 | Two-stage training |
| `mixed` | Combined Data | Contrastive | 48 | Reddit + Internal |
| `regularized` | Internal Only | Contrastive | 16 | Anti-overfitting |

### Key Configuration Parameters

```python
# Model architecture
"style_feature_dim": 1169,      # Actual extracted dimensions
"hidden_dim": 768,              # Encoder capacity
"final_embedding_dim": 256,     # Output embedding size
"dropout_rate": 0.2,            # Regularization

# Training parameters
"learning_rate": 2e-4,          # Optimized for large batches
"batch_size": 256,              # 4x GPU optimization
"num_epochs": 15,               # Fewer epochs with more data
"margin": 0.5,                  # Contrastive loss margin

# Two-stage training
"training_mode": "two_stage",   # Options: classification, verification, two_stage
"classification_epochs": 10,    # Classification pretraining epochs
"verification_epochs": 15,      # Verification fine-tuning epochs

# Data settings
"max_features": 1000,           # Stylometric feature limit
"min_text_length": 100,         # Minimum sample length
"validation_split": 0.2         # Internal validation split
```

## Baseline Model Details

### LUAR (Learning Universal Authorship Representations)
- **Architecture**: Transformer-based with episodic processing
- **Input**: 16 text episodes per batch, 32 token limit
- **Output**: 512-dimensional embeddings
- **Compatibility**: CPU-only (MPS compatibility issues)
- **Performance**: 61.43% accuracy, 79.06% AUC

### ModernBERT (Authorship Verification)
- **Architecture**: Modern BERT variant with sentence transformers
- **Input**: Up to 8,192 tokens per text
- **Output**: 768-dimensional embeddings
- **Compatibility**: Full MPS/CUDA support
- **Performance**: 65.00% accuracy, 75.82% AUC

## Technical Requirements

### Local Development
- **Python**: 3.13+ (modern typing support)
- **PyTorch**: 2.5.0+ with MPS support
- **Dependencies**: sentence-transformers, transformers, einops
- **Database**: PostgreSQL access to decide_development
- **Memory**: 8GB+ RAM, 2GB+ GPU memory

### Vertex AI Training
- **Project**: ${GCP_PROJECT}
- **Service Account**: vertex-training-sa@${GCP_PROJECT}.iam.gserviceaccount.com
- **Storage**: GCS bucket `${GCS_BUCKET}` (us-central1)
- **Compute**: n1-standard-8 with NVIDIA T4 GPU
- **Docker**: gcr.io/${GCP_PROJECT}/writing-identification:v38

### Dependencies
```bash
# Production training (included in Docker)
torch>=2.5.0
sentence-transformers>=3.1.0
scikit-learn>=1.5.2
datasets>=2.14.0
google-cloud-storage>=2.10.0

# Local development only (excluded from Docker)
transformers>=4.36.0  # For LUAR baseline
einops>=0.8.0        # For LUAR model architecture
```

## Performance Analysis

### Confusion Matrix Interpretation

The confusion matrices show model behavior patterns:
- **True Positives**: Correctly identified same-author pairs
- **True Negatives**: Correctly identified different-author pairs
- **False Positives**: Incorrectly flagged different authors as same (Type I error)
- **False Negatives**: Incorrectly flagged same author as different (Type II error)

**Threshold Optimization**: Models use optimal thresholds from validation data (custom model) or calibration set (pre-trained models) to avoid symmetric confusion matrices and data leakage.

### Error Analysis Capabilities
- **Sensitivity (Recall)**: How well models detect same-author pairs
- **Specificity**: How well models detect different-author pairs
- **Balanced Performance**: Trade-off between sensitivity and specificity
- **Error Patterns**: Understand where baseline models fail

## Output Files

### Training Outputs
- `results/models/best_model.pt` - Best performing model checkpoint with saved threshold
- `results/models/checkpoint_epoch_*.pt` - Regular epoch checkpoints
- `results/training_history_*.json` - Complete training metrics
- `cache/features_*.pkl` - Cached feature extractions
- `cache/reddit_pairs/` - Cached Reddit pair manifests

### Baseline Evaluation Outputs
- `baselines/results/baseline_evaluation_*.json` - Detailed metrics and confusion matrices
- `baselines/results/baseline_test.log` - Evaluation logs
- Programmatic access to all performance data

### GCS Integration (Vertex AI)
- `gs://${GCS_BUCKET}/models/` - Model checkpoints
- `gs://${GCS_BUCKET}/results/` - Training results
- `gs://${GCS_BUCKET}/datasets/` - Cached Reddit datasets

## Usage Notes

### Performance Optimization
- **Pre-computed Features**: 1200x+ speedup with HDF5-based feature pre-computation
- **MPS Acceleration**: Automatic on Apple Silicon for compatible models
- **Large Batch Sizes**: 4x optimization based on GPU memory analysis
- **Feature Caching**: Automatic caching for 5-10x faster subsequent runs
- **GCS Caching**: Reddit datasets cached for fast Vertex AI startup
- **Memory Efficiency**: Pre-computed features reduce GPU memory usage

### Compatibility
- **Multiprocessing**: Disabled for MPS compatibility
- **LUAR Fallback**: CPU-only due to MPS tensor issues
- **Docker Exclusions**: Baseline models excluded from production builds
- **Device Detection**: Automatic CUDA/MPS/CPU selection

### Model Deployment
- **Local Testing**: Full baseline comparison framework
- **Production Training**: Scalable Vertex AI with GPU acceleration
- **Experiment Tracking**: JSON-based results for analysis
- **Early Stopping**: Automatic overfitting prevention

## Next Steps

1. **Custom Model Development**: Continue training Siamese network variants with more epochs
2. **Baseline Comparison**: Evaluate custom models against LUAR/ModernBERT benchmarks
3. **Error Analysis**: Deep-dive into confusion matrices for model improvement
4. **Hyperparameter Optimization**: Systematic tuning based on baseline insights
5. **Production Deployment**: Scale best-performing models for real-world usage
