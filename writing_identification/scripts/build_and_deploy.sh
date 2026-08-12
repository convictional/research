#!/bin/bash

# Build and deploy script for Vertex AI training container

set -e  # Exit on error

# Configuration
PROJECT_ID=${GCP_PROJECT_ID:-""}
IMAGE_NAME=${IMAGE_NAME:-"writing-identification"}
IMAGE_TAG=${IMAGE_TAG:-"latest"}
REGION=${REGION:-"us-central1"}
REPOSITORY=${REPOSITORY:-"gcr.io"}  # Can be gcr.io or [REGION]-docker.pkg.dev

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Check required environment variables
check_requirements() {
    print_message "$YELLOW" "Checking requirements..."

    if [ -z "$PROJECT_ID" ]; then
        print_message "$RED" "Error: GCP_PROJECT_ID environment variable is not set"
        echo "Please set it with: export GCP_PROJECT_ID=your-project-id"
        exit 1
    fi

    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        print_message "$RED" "Error: gcloud CLI is not installed"
        echo "Please install it from: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi

    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        print_message "$RED" "Error: Docker is not installed"
        echo "Please install Docker Desktop"
        exit 1
    fi

    print_message "$GREEN" "✓ All requirements met"
}

# Configure Docker for GCR
configure_docker() {
    print_message "$YELLOW" "Configuring Docker for Google Container Registry..."

    # Configure docker to use gcloud for authentication
    gcloud auth configure-docker ${REPOSITORY} --quiet

    print_message "$GREEN" "✓ Docker configured"
}

# Build Docker image
build_image() {
    local full_image_name="${REPOSITORY}/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}"

    print_message "$YELLOW" "Building Docker image: ${full_image_name}"

    # Build the image
    docker build -t ${full_image_name} .

    if [ $? -eq 0 ]; then
        print_message "$GREEN" "✓ Docker image built successfully"
    else
        print_message "$RED" "✗ Docker build failed"
        exit 1
    fi
}

# Push Docker image to GCR
push_image() {
    local full_image_name="${REPOSITORY}/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}"

    print_message "$YELLOW" "Pushing Docker image to ${REPOSITORY}..."

    # Push the image
    docker push ${full_image_name}

    if [ $? -eq 0 ]; then
        print_message "$GREEN" "✓ Docker image pushed successfully"
        print_message "$GREEN" "Image URI: ${full_image_name}"
    else
        print_message "$RED" "✗ Docker push failed"
        exit 1
    fi
}

# Optional: Run a test container locally
test_locally() {
    local full_image_name="${REPOSITORY}/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}"

    print_message "$YELLOW" "Testing container locally..."

    # Run container with test environment variables
    docker run --rm \
        -e VERTEX_AI_ENABLED=false \
        -e GCP_PROJECT=${PROJECT_ID} \
        -e EXPERIMENT_NAME=local_test \
        ${full_image_name} \
        vertex_train --mode train
}

# Main execution
main() {
    print_message "$GREEN" "========================================="
    print_message "$GREEN" "Vertex AI Training Container Build & Deploy"
    print_message "$GREEN" "========================================="

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --project-id)
                PROJECT_ID="$2"
                shift 2
                ;;
            --image-name)
                IMAGE_NAME="$2"
                shift 2
                ;;
            --tag)
                IMAGE_TAG="$2"
                shift 2
                ;;
            --test)
                TEST_LOCALLY=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --project-id STRING   GCP Project ID (or set GCP_PROJECT_ID env var)"
                echo "  --image-name STRING   Docker image name (default: writing-identification)"
                echo "  --tag STRING          Docker image tag (default: latest)"
                echo "  --test                Run a test container locally after building"
                echo "  --help                Show this help message"
                exit 0
                ;;
            *)
                print_message "$RED" "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done

    # Execute build steps
    check_requirements
    configure_docker
    build_image
    push_image

    if [ "$TEST_LOCALLY" = true ]; then
        test_locally
    fi

    print_message "$GREEN" "========================================="
    print_message "$GREEN" "Build and deploy completed successfully!"
    print_message "$GREEN" "========================================="

    # Print next steps
    echo ""
    print_message "$YELLOW" "Next steps:"
    echo "1. Submit a training job:"
    echo "   python scripts/submit_vertex_job.py \\"
    echo "     --project-id ${PROJECT_ID} \\"
    echo "     --image-uri ${REPOSITORY}/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG} \\"
    echo "     --gcs-bucket YOUR_BUCKET \\"
    echo "     --experiment-name YOUR_EXPERIMENT"
    echo ""
    echo "2. For hyperparameter search:"
    echo "   python scripts/submit_vertex_job.py \\"
    echo "     --project-id ${PROJECT_ID} \\"
    echo "     --image-uri ${REPOSITORY}/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG} \\"
    echo "     --job-type hyperparam \\"
    echo "     --gcs-bucket YOUR_BUCKET"
}

# Run main function
main "$@"
