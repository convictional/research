"""Submit training jobs to Vertex AI."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
import yaml

from google.cloud import aiplatform

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
try:
    from config.experiment_configs import get_experiment_config, list_experiment_configs
except ImportError:
    # Fallback if experiment_configs.py doesn't exist
    get_experiment_config = None
    list_experiment_configs = None


class VertexJobSubmitter:
    """Handles submission of training jobs to Vertex AI."""

    def __init__(self, project_id: str, region: str = "us-central1", staging_bucket: str = None):
        """
        Initialize job submitter.

        Args:
            project_id: GCP project ID
            region: GCP region for Vertex AI
            staging_bucket: GCS bucket for staging
        """
        self.project_id = project_id
        self.region = region
        self.staging_bucket = staging_bucket

        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=region, staging_bucket=staging_bucket)

    def submit_single_training_job(
        self,
        display_name: str,
        container_image_uri: str,
        machine_type: str = "n1-highmem-8",
        accelerator_type: str = "NVIDIA_TESLA_T4",
        accelerator_count: int = 1,
        environment_variables: dict[str, str] = None,
        args: list[str] = None,
        replica_count: int = 1,
        boot_disk_size_gb: int = 100,
    ) -> aiplatform.CustomJob:
        """
        Submit a single training job to Vertex AI.

        Args:
            display_name: Job display name
            container_image_uri: Docker image URI
            machine_type: Machine type (e.g., n1-standard-8)
            accelerator_type: GPU type (e.g., NVIDIA_TESLA_T4)
            accelerator_count: Number of GPUs
            environment_variables: Environment variables for the container
            args: Command line arguments
            replica_count: Number of training replicas
            boot_disk_size_gb: Boot disk size in GB

        Returns:
            Submitted CustomJob object
        """
        # Prepare worker pool specs
        worker_pool_specs = [
            {
                "machine_spec": {
                    "machine_type": machine_type,
                    "accelerator_type": accelerator_type,
                    "accelerator_count": accelerator_count,
                },
                "replica_count": replica_count,
                "container_spec": {
                    "image_uri": container_image_uri,
                    "env": [{"name": k, "value": v} for k, v in (environment_variables or {}).items()],
                    "args": args or [],
                },
                "disk_spec": {
                    "boot_disk_type": "pd-ssd",
                    "boot_disk_size_gb": boot_disk_size_gb,
                },
            }
        ]

        # Create and submit job
        job = aiplatform.CustomJob(
            display_name=display_name,
            worker_pool_specs=worker_pool_specs,
            project=self.project_id,
            location=self.region,
        )

        # Use custom service account for proper GCS access
        job.run(service_account="vertex-training-sa@${GCP_PROJECT}.iam.gserviceaccount.com", sync=True)

        print(f"Job submitted: {job.display_name}")
        print(f"Job resource name: {job.resource_name}")

        return job

    def submit_hyperparameter_tuning_job(
        self,
        display_name: str,
        container_image_uri: str,
        hyperparameter_config: dict,
        metric_spec: dict,
        max_trial_count: int = 20,
        parallel_trial_count: int = 4,
        machine_type: str = "n1-highmem-8",
        accelerator_type: str = "NVIDIA_TESLA_T4",
        accelerator_count: int = 1,
        environment_variables: dict[str, str] = None,
    ) -> aiplatform.HyperparameterTuningJob:
        """
        Submit a hyperparameter tuning job to Vertex AI.

        Args:
            display_name: Job display name
            container_image_uri: Docker image URI
            hyperparameter_config: Hyperparameter search configuration
            metric_spec: Metric to optimize (e.g., {"metric_id": "val_auc", "goal": "MAXIMIZE"})
            max_trial_count: Maximum number of trials
            parallel_trial_count: Number of parallel trials
            machine_type: Machine type
            accelerator_type: GPU type
            accelerator_count: Number of GPUs
            environment_variables: Base environment variables

        Returns:
            Submitted HyperparameterTuningJob object
        """
        # Define parameter specs from config
        parameter_spec = {}
        for param_name, param_config in hyperparameter_config.items():
            if "values" in param_config:
                # Categorical parameter
                parameter_spec[param_name] = aiplatform.hyperparameter_tuning.CategoricalParameterSpec(
                    values=param_config["values"]
                )
            elif "min" in param_config and "max" in param_config:
                # Continuous parameter
                scale = param_config.get("scale", "linear")
                if param_config.get("type") == "integer":
                    parameter_spec[param_name] = aiplatform.hyperparameter_tuning.IntegerParameterSpec(
                        min=param_config["min"], max=param_config["max"], scale=scale
                    )
                else:
                    parameter_spec[param_name] = aiplatform.hyperparameter_tuning.DoubleParameterSpec(
                        min=param_config["min"], max=param_config["max"], scale=scale
                    )

        # Create custom job spec
        custom_job_spec = {
            "worker_pool_specs": [
                {
                    "machine_spec": {
                        "machine_type": machine_type,
                        "accelerator_type": accelerator_type,
                        "accelerator_count": accelerator_count,
                    },
                    "replica_count": 1,
                    "container_spec": {
                        "image_uri": container_image_uri,
                        "env": [{"name": k, "value": v} for k, v in (environment_variables or {}).items()],
                    },
                }
            ]
        }

        # Create hyperparameter tuning job
        hp_job = aiplatform.HyperparameterTuningJob(
            display_name=display_name,
            custom_job=custom_job_spec,
            metric_spec=metric_spec,
            parameter_spec=parameter_spec,
            max_trial_count=max_trial_count,
            parallel_trial_count=parallel_trial_count,
            project=self.project_id,
            location=self.region,
        )

        # Use custom service account for proper GCS access
        hp_job.run(service_account="vertex-training-sa@${GCP_PROJECT}.iam.gserviceaccount.com", sync=True)

        print(f"Hyperparameter tuning job submitted: {hp_job.display_name}")
        print(f"Job resource name: {hp_job.resource_name}")

        return hp_job

    def submit_batch_jobs(
        self,
        configs: list[dict],
        base_display_name: str,
        container_image_uri: str,
        machine_type: str = "n1-highmem-8",
        accelerator_type: str = "NVIDIA_TESLA_T4",
        accelerator_count: int = 1,
        base_env_vars: dict[str, str] = None,
    ) -> list[aiplatform.CustomJob]:
        """
        Submit multiple training jobs with different configurations.

        Args:
            configs: list of configuration override dictionaries
            base_display_name: Base name for jobs
            container_image_uri: Docker image URI
            machine_type: Machine type
            accelerator_type: GPU type
            accelerator_count: Number of GPUs
            base_env_vars: Base environment variables

        Returns:
            list of submitted CustomJob objects
        """
        jobs = []

        for i, config in enumerate(configs):
            # Prepare environment variables
            env_vars = base_env_vars.copy() if base_env_vars else {}
            env_vars["CONFIG_OVERRIDES"] = json.dumps(config)
            env_vars["RUN_ID"] = f"{base_display_name}_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Submit job
            job_name = f"{base_display_name}_config_{i}"
            job = self.submit_single_training_job(
                display_name=job_name,
                container_image_uri=container_image_uri,
                machine_type=machine_type,
                accelerator_type=accelerator_type,
                accelerator_count=accelerator_count,
                environment_variables=env_vars,
            )

            jobs.append(job)

        print(f"Submitted {len(jobs)} batch jobs")
        return jobs


def main():
    """Main function for job submission."""
    parser = argparse.ArgumentParser(description="Submit Vertex AI training jobs")

    parser.add_argument("--project-id", type=str, required=True, help="GCP project ID")

    parser.add_argument("--region", type=str, default="us-central1", help="GCP region for Vertex AI")

    parser.add_argument(
        "--image-uri",
        type=str,
        required=True,
        help="Docker image URI (e.g., gcr.io/PROJECT/writing-identification:latest)",
    )

    parser.add_argument(
        "--job-type",
        type=str,
        choices=["single", "hyperparam", "batch"],
        default="single",
        help="Type of job to submit",
    )

    parser.add_argument(
        "--display-name",
        type=str,
        default=f"authorship-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        help="Job display name",
    )

    parser.add_argument("--machine-type", type=str, default="n1-highmem-8", help="Machine type")

    parser.add_argument("--accelerator-type", type=str, default="NVIDIA_TESLA_T4", help="GPU type")

    parser.add_argument("--accelerator-count", type=int, default=1, help="Number of GPUs")

    parser.add_argument("--gcs-bucket", type=str, required=True, help="GCS bucket for artifacts")

    parser.add_argument("--experiment-name", type=str, default="default", help="Experiment name")

    parser.add_argument("--config-file", type=str, help="Path to configuration file (JSON or YAML)")

    parser.add_argument("--config-overrides", type=str, help="JSON string of config overrides")

    parser.add_argument(
        "--experiment-config",
        type=str,
        help="Predefined experiment configuration name (use --list-experiments to see available)",
    )

    parser.add_argument(
        "--list-experiments", action="store_true", help="list available predefined experiment configurations"
    )

    args = parser.parse_args()

    # Handle listing experiments
    if args.list_experiments:
        if list_experiment_configs:
            configs = list_experiment_configs()
            print("\n📋 Available predefined experiment configurations:")
            for name, description in configs.items():
                print(f"   {name}: {description}")
        else:
            print("⚠️  Experiment configurations not available (experiment_configs.py not found)")
        return

    # Initialize submitter
    submitter = VertexJobSubmitter(
        project_id=args.project_id, region=args.region, staging_bucket=f"gs://{args.gcs_bucket}/staging"
    )

    # Prepare base environment variables
    base_env_vars = {
        "GCP_PROJECT": args.project_id,
        "GCS_BUCKET": args.gcs_bucket,
        "EXPERIMENT_NAME": args.experiment_name,
        "VERTEX_AI_ENABLED": "true",
    }

    # Handle experiment configuration
    config_overrides_json = None
    if args.experiment_config:
        if get_experiment_config:
            try:
                experiment_config = get_experiment_config(args.experiment_config)
                config_overrides_json = json.dumps(experiment_config)
                print(f"\n🧪 Using experiment configuration: {args.experiment_config}")
                print(f"   Config: {json.dumps(experiment_config, indent=2)}")
            except ValueError as e:
                print(f"❌ Error loading experiment config: {e}")
                return
        else:
            print("⚠️  Experiment configurations not available (experiment_configs.py not found)")
            return
    elif args.config_overrides:
        config_overrides_json = args.config_overrides

    # Add config overrides if provided
    if config_overrides_json:
        base_env_vars["CONFIG_OVERRIDES"] = config_overrides_json

    # Submit job based on type
    if args.job_type == "single":
        # Submit single training job
        _job = submitter.submit_single_training_job(
            display_name=args.display_name,
            container_image_uri=args.image_uri,
            machine_type=args.machine_type,
            accelerator_type=args.accelerator_type,
            accelerator_count=args.accelerator_count,
            environment_variables=base_env_vars,
        )

    elif args.job_type == "hyperparam":
        # Load hyperparameter configuration
        if args.config_file:
            with open(args.config_file, "r") as f:
                if args.config_file.endswith(".yaml"):
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
        else:
            # Default hyperparameter search config
            config = {
                "learning_rate": {"min": 1e-5, "max": 1e-3, "scale": "log"},
                "batch_size": {"values": [16, 32, 64]},
                "margin": {"min": 0.1, "max": 0.5},
                "dropout_rate": {"min": 0.1, "max": 0.4},
            }

        # Submit hyperparameter tuning job
        _job = submitter.submit_hyperparameter_tuning_job(
            display_name=args.display_name,
            container_image_uri=args.image_uri,
            hyperparameter_config=config,
            metric_spec={"metric_id": "val_auc", "goal": "MAXIMIZE"},
            machine_type=args.machine_type,
            accelerator_type=args.accelerator_type,
            accelerator_count=args.accelerator_count,
            environment_variables=base_env_vars,
        )

    elif args.job_type == "batch":
        # Load batch configurations
        if not args.config_file:
            raise ValueError("Config file required for batch jobs")

        with open(args.config_file, "r") as f:
            if args.config_file.endswith(".yaml"):
                configs = yaml.safe_load(f)
            else:
                configs = json.load(f)

        # Submit batch jobs
        _jobs = submitter.submit_batch_jobs(
            configs=configs,
            base_display_name=args.display_name,
            container_image_uri=args.image_uri,
            machine_type=args.machine_type,
            accelerator_type=args.accelerator_type,
            accelerator_count=args.accelerator_count,
            base_env_vars=base_env_vars,
        )

    print("Job submission completed!")


if __name__ == "__main__":
    main()
