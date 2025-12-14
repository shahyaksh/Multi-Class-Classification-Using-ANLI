#!/bin/bash

set -e  # Exit on error

echo "=========================================="
echo "ANLI NLI Model - Cloud Run Deployment"
echo "=========================================="
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "ERROR: gcloud CLI not found. Please install it first:"
    echo "https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Get configuration
read -p "Enter your Google Cloud Project ID: " PROJECT_ID
read -p "Enter your W&B API key: " WANDB_API_KEY
read -p "Enter region (default: us-central1): " REGION
REGION=${REGION:-us-central1}

# Set variables
IMAGE_NAME="deberta-anli-nli"
SERVICE_NAME="deberta-anli-nli"

echo ""
echo "Configuration:"
echo "  Project ID: $PROJECT_ID"
echo "  Region: $REGION"
echo "  Image: gcr.io/$PROJECT_ID/$IMAGE_NAME"
echo "  W&B API Key: ${WANDB_API_KEY:0:10}..."
echo ""

read -p "Continue with deployment? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled."
    exit 0
fi

# Set project
echo ""
echo "Step 1: Setting Google Cloud project..."
gcloud config set project $PROJECT_ID

# Enable APIs
echo ""
echo "Step 2: Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build image
echo ""
echo "Step 3: Building Docker image with model..."
gcloud builds submit --config=cloudbuild.yaml \
  --substitutions=_WANDB_API_KEY=$WANDB_API_KEY

# Deploy to Cloud Run
echo ""
echo "Step 4: Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$IMAGE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 4 \
  --timeout 300 \
  --max-instances 10 \
  --min-instances 0 \
  --concurrency 10

# Get service URL
echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""

SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format='value(status.url)')

echo "Your API is now live at:"
echo "  $SERVICE_URL"
echo ""
echo "Test endpoints:"
echo "  Health: $SERVICE_URL/health"
echo "  Docs: $SERVICE_URL/docs"
echo "  Predict: $SERVICE_URL/predict"
echo ""
echo "Example curl command:"
echo "curl -X POST $SERVICE_URL/predict \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"premise\": \"A dog is running\", \"hypothesis\": \"An animal is moving\"}'"
echo ""
echo "Note: First request may take 30-60 seconds as the model downloads from W&B."
echo "Subsequent requests will be fast!"
