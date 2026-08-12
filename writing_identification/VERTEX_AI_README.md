# Vertex AI Training Pipeline

This document describes how to run the authorship verification training pipeline on Google Cloud Vertex AI for massively parallel experimentation.

## 🚀 Quick Start

```bash
# 1. Set up environment
export GCP_PROJECT_ID="your-project-id"
export GCS_BUCKET="your-bucket-name"

# 2. Build and push Docker image
./scripts/build_and_deploy.sh --project-id $GCP_PROJECT_ID

# 3. Submit a training job
python scripts/submit_vertex_job.py \
  --project-id $GCP_PROJECT_ID \
  --image-uri gcr.io/$GCP_PROJECT_ID/writing-identification:latest \
  --gcs-bucket $GCS_BUCKET \
  --experiment-name "baseline_experiment"
```

## 📁 Project Structure

```
writing_identification/
├── Dockerfile                    # GPU-enabled container definition
├── requirements.txt             # Python dependencies for container
├── vertex_train.py             # Main training entrypoint for Vertex AI
├── config/
│   ├── config.py               # Base experiment configuration
│   └── vertex_config.py        # Vertex AI configuration system
├── utils/
│   └── gcs_storage.py          # GCS storage adapter for artifacts
└── scripts/
    ├── build_and_deploy.sh     # Docker build & push script
    └── submit_vertex_job.py    # Vertex AI job submission
```

## 🛠️ Setup

### Prerequisites

1. **Google Cloud SDK**
   ```bash
   # Install gcloud CLI
   curl https://sdk.cloud.google.com | bash
   gcloud auth login
   gcloud config set project $GCP_PROJECT_ID
   ```

2. **Enable APIs**
   ```bash
   gcloud services enable \
     aiplatform.googleapis.com \
     storage.googleapis.com \
     cloudbuild.googleapis.com \
     containerregistry.googleapis.com
   ```

3. **Create GCS Bucket**
   ```bash
   gcloud storage buckets create gs://$GCS_BUCKET --project=$GCP_PROJECT_ID
   ```

4. **Docker Authentication**
   ```bash
   gcloud auth configure-docker
   ```

## 🏗️ Building & Deploying

### Build Docker Image

```bash
# Basic build
./scripts/build_and_deploy.sh --project-id $GCP_PROJECT_ID

# With custom tag
./scripts/build_and_deploy.sh \
  --project-id $GCP_PROJECT_ID \
  --tag v1.0.0

# Test locally first
./scripts/build_and_deploy.sh \
  --project-id $GCP_PROJECT_ID \
  --test
```

The image will be pushed to: `gcr.io/$GCP_PROJECT_ID/writing-identification:latest`

## 🚂 Training Jobs

### Single Training Run

```bash
python scripts/submit_vertex_job.py \
  --project-id $GCP_PROJECT_ID \
  --image-uri gcr.io/$GCP_PROJECT_ID/writing-identification:latest \
  --gcs-bucket $GCS_BUCKET \
  --experiment-name "experiment_1" \
  --machine-type "n1-standard-8" \
  --accelerator-type "NVIDIA_TESLA_T4" \
  --config-overrides '{"training": {"batch_size": 64, "learning_rate": 1e-4}}'
```

### Hyperparameter Search

Create a hyperparameter config file `hyperparam_config.json`:

```json
{
  "learning_rate": {"min": 1e-5, "max": 1e-3, "scale": "log"},
  "batch_size": {"values": [16, 32, 64, 128]},
  "margin": {"min": 0.1, "max": 0.5},
  "dropout_rate": {"min": 0.1, "max": 0.4},
  "hidden_dim": {"values": [256, 512, 768]}
}
```

Submit the hyperparameter search:

```bash
python scripts/submit_vertex_job.py \
  --project-id $GCP_PROJECT_ID \
  --image-uri gcr.io/$GCP_PROJECT_ID/writing-identification:latest \
  --job-type hyperparam \
  --gcs-bucket $GCS_BUCKET \
  --config-file hyperparam_config.json \
  --display-name "hp_search_$(date +%Y%m%d)"
```

### Batch Experiments

Create a batch config file `batch_configs.json`:

```json
[
  {"model": {"encoder_type": "fusion"}, "training": {"learning_rate": 1e-5}},
  {"model": {"encoder_type": "attention"}, "training": {"learning_rate": 5e-5}},
  {"model": {"encoder_type": "simple"}, "training": {"learning_rate": 1e-4}}
]
```

Submit batch jobs:

```bash
python scripts/submit_vertex_job.py \
  --project-id $GCP_PROJECT_ID \
  --image-uri gcr.io/$GCP_PROJECT_ID/writing-identification:latest \
  --job-type batch \
  --gcs-bucket $GCS_BUCKET \
  --config-file batch_configs.json \
  --display-name "batch_experiment"
```

## 🖥️ Machine Types & GPUs

### Recommended Configurations

| Use Case | Machine Type | GPU | Memory | Cost/hr* |
|----------|-------------|-----|---------|----------|
| Development | n1-standard-4 | T4 | 15GB | ~$0.40 |
| Standard Training | n1-standard-8 | T4 | 30GB | ~$0.50 |
| Large Batch | n1-highmem-8 | V100 | 52GB | ~$2.50 |
| Production | a2-highgpu-1g | A100 | 85GB | ~$3.70 |

*Approximate costs in us-central1 region

### Setting Machine Type

```bash
# Development with T4
--machine-type "n1-standard-4" \
--accelerator-type "NVIDIA_TESLA_T4"

# Production with A100
--machine-type "a2-highgpu-1g" \
--accelerator-type "NVIDIA_TESLA_A100"
```

## 📊 Monitoring & Results

### View Job Status

```bash
# List recent jobs
gcloud ai custom-jobs list --region=us-central1

# Stream logs for a specific job
gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1
```

### Access Results in GCS

```bash
# List experiments
gcloud storage ls gs://$GCS_BUCKET/writing-identification/

# Download results
gcloud storage cp -r gs://$GCS_BUCKET/writing-identification/EXPERIMENT_NAME/results .

# Get best model
gcloud storage cp gs://$GCS_BUCKET/writing-identification/EXPERIMENT_NAME/models/best_model.pt .
```

### TensorBoard Integration

```bash
# Create TensorBoard instance
gcloud ai tensorboards create \
  --display-name "authorship-tensorboard" \
  --region=us-central1

# Get TensorBoard ID
TB_RESOURCE=$(gcloud ai tensorboards list --region=us-central1 --format="value(name)")

# Use in training
--environment-variables "TENSORBOARD_RESOURCE=$TB_RESOURCE"
```

## 🔧 Configuration

### Environment Variables

The training container accepts these environment variables:

| Variable | Description | Default |
|----------|------------|---------|
| `GCP_PROJECT` | GCP Project ID | Required |
| `GCS_BUCKET` | GCS bucket for artifacts | Required |
| `EXPERIMENT_NAME` | Experiment identifier | "default" |
| `VERTEX_AI_ENABLED` | Enable Vertex AI features | "true" |
| `CONFIG_OVERRIDES` | JSON config overrides | "{}" |
| `USE_TENSORBOARD` | Enable TensorBoard logging | "true" |
| `USE_WANDB` | Enable Weights & Biases | "false" |

### Config Overrides

Override any configuration parameter via `CONFIG_OVERRIDES`:

```json
{
  "training": {
    "batch_size": 64,
    "learning_rate": 1e-4,
    "num_epochs": 100,
    "early_stopping_patience": 15
  },
  "model": {
    "hidden_dim": 768,
    "dropout_rate": 0.3,
    "encoder_type": "attention"
  }
}
```

## 🐛 Troubleshooting

### Common Issues

1. **Docker push fails**
   ```bash
   # Re-authenticate Docker
   gcloud auth configure-docker
   ```

2. **Out of Memory**
   - Reduce batch size
   - Use larger machine type
   - Enable gradient checkpointing

3. **Job fails immediately**
   ```bash
   # Check logs
   gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1
   ```

4. **Slow training**
   - Ensure GPU is being utilized
   - Check data loading bottlenecks
   - Consider using larger batch sizes

## 💰 Cost Optimization

1. **Use Preemptible VMs** (70% discount)
   ```python
   # In submit_vertex_job.py, add to worker_pool_specs:
   "scheduling": {"disable_retries": False}
   ```

2. **Use Spot VMs** (60-91% discount)
   - Good for hyperparameter search
   - May be preempted

3. **Regional Pricing**
   - us-central1 is typically cheapest
   - Check current pricing: https://cloud.google.com/vertex-ai/pricing

4. **Efficient Hyperparameter Search**
   - Use Bayesian optimization instead of grid search
   - Set reasonable `max_trial_count` and `parallel_trial_count`

## 📈 Advanced Features

### Multi-GPU Training

```python
# Modify Dockerfile to support distributed training
ENV NCCL_DEBUG=INFO
ENV TORCH_DISTRIBUTED_DEBUG=DETAIL

# In vertex_train.py, add distributed training support
```

### Custom Metrics

```python
# In vertex_train.py
aiplatform.log_metrics({
    "custom_metric": value,
    "epoch": epoch
})
```

### Model Registry

```python
# After training, register model
model = aiplatform.Model.upload(
    display_name="authorship-model",
    artifact_uri=f"gs://{bucket}/models/",
    serving_container_image_uri="..."
)
```

## 📚 Additional Resources

- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Custom Training Guide](https://cloud.google.com/vertex-ai/docs/training/custom-training)
- [Hyperparameter Tuning](https://cloud.google.com/vertex-ai/docs/training/hyperparameter-tuning-overview)
- [GPU Selection Guide](https://cloud.google.com/compute/docs/gpus)

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review job logs in Cloud Console
3. File an issue in the repository
