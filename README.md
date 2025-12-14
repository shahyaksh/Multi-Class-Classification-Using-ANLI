# Google Cloud Run Deployment Guide

Deploy your fine-tuned DeBERTa ANLI model to Google Cloud Run.

## Prerequisites

1. **Google Cloud Account** with billing enabled
2. **gcloud CLI** installed: https://cloud.google.com/sdk/docs/install
3. **W&B API Key**: Get it from https://wandb.ai/authorize

---

## Quick Start

### 1. Get Your W&B API Key

1. Go to https://wandb.ai/authorize
2. Copy your API key
3. Save it for the deployment

### 2. Set Up Google Cloud

```bash
# Login to Google Cloud
gcloud auth login

# Set your project ID
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

### 3. Deploy Using the Script

```bash
cd deployment/cloud-run
./deploy.sh
```

The script will prompt you for:
- Google Cloud Project ID
- W&B API key
- Region (default: us-central1)

**Deployment time**: ~12-15 minutes (first time), ~5-7 minutes (subsequent)

---

## Manual Deployment

If you prefer to deploy manually:

### Build the Image

```bash
export PROJECT_ID="your-project-id"
export WANDB_API_KEY="your-wandb-api-key"

gcloud builds submit --config=cloudbuild.yaml \
  --substitutions=_WANDB_API_KEY=$WANDB_API_KEY
```

### Deploy to Cloud Run

```bash
gcloud run deploy deberta-anli-nli \
  --image gcr.io/$PROJECT_ID/deberta-anli-nli \
  --platform managed \
  --region us-east1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 4 \
  --timeout 300 \
  --max-instances 10 \
  --min-instances 0 \
  --concurrency 10
```

---

## Testing Your Deployment

After deployment, you'll get a service URL. Test it:

```bash
export SERVICE_URL="https://deberta-anli-nli-66725207998.us-east1.run.app"

# Health check
curl $SERVICE_URL/health

# Single prediction
curl -X POST $SERVICE_URL/predict \
  -H "Content-Type: application/json" \
  -d '{
    "premise": "A person is walking a dog in the park",
    "hypothesis": "A person is outside"
  }'
```

**Expected Response**:
```json
{
  "prediction": "Entailment",
  "confidence": 0.95,
  "probabilities": {
    "entailment": 0.95,
    "neutral": 0.04,
    "contradiction": 0.01
  }
}
```

### Python Example

```python
import requests

SERVICE_URL = "https://deberta-anli-nli-66725207998.us-east1.run.app"

response = requests.post(
    f"{SERVICE_URL}/predict",
    json={
        "premise": "A person is walking a dog in the park",
        "hypothesis": "A person is outside"
    }
)

print(response.json())
```

---

## API Documentation

Once deployed, visit:
- **Interactive Docs**: `https://your-service-url/docs`
- **ReDoc**: `https://your-service-url/redoc`

---

## Configuration Options

### Memory & CPU

```bash
--memory 4Gi      # 2Gi, 4Gi, 8Gi
--cpu 4           # 1, 2, 4, 8
```

### Scaling

```bash
--min-instances 0    # Scale to zero (no idle costs)
--max-instances 10   # Maximum concurrent instances
--concurrency 10     # Requests per instance
```

### Timeout

```bash
--timeout 300        # 300 seconds (5 minutes)
```

---

## Monitoring

### View Logs

```bash
gcloud run services logs read deberta-anli-nli --region us-east1 --limit 50
```

### Update Deployment

After making changes to `main.py`:

```bash
# Rebuild
gcloud builds submit --config=cloudbuild.yaml \
  --substitutions=_WANDB_API_KEY=$WANDB_API_KEY

# Redeploy
gcloud run deploy deberta-anli-nli \
  --image gcr.io/$PROJECT_ID/deberta-anli-nli \
  --region us-east1
```

---

## Troubleshooting

### Build Fails

**Check logs**:
```bash
gcloud builds log --limit=50
```

**Common issues**:
- W&B API key incorrect
- Insufficient permissions
- Network timeout

### Container Fails to Start

**Check logs**:
```bash
gcloud run services logs read deberta-anli-nli --region us-east1
```

**Common fixes**:
- Increase memory: `--memory 8Gi`
- Increase timeout: `--timeout 600`
- Check model path in `main.py`

---

## Cost Estimate

Cloud Run pricing (pay per use):

| Traffic | Requests/Day | Monthly Cost |
|---------|--------------|--------------|
| Low | 100 | ~$2-3 |
| Medium | 1,000 | ~$15-20 |
| High | 10,000 | ~$150-200 |

**Free Tier**: 2M requests/month

**Note**: Scales to zero when not in use = no idle costs!

---

## Clean Up

To delete the deployment:

```bash
# Delete service
gcloud run services delete deberta-anli-nli --region us-east1

# Delete images
gcloud container images delete gcr.io/$PROJECT_ID/deberta-anli-nli
```

---

## Files Overview

- `main.py` - FastAPI application
- `Dockerfile` - Container configuration
- `download_model.py` - W&B model download script
- `cloudbuild.yaml` - Cloud Build configuration
- `deploy.sh` - Automated deployment script
- `requirements.txt` - Python dependencies
- `test_api.py` - API testing script

---

## Support

For issues:
1. Check logs: `gcloud run services logs read deberta-anli-nli`
2. Review [Cloud Run documentation](https://cloud.google.com/run/docs)
3. Check [troubleshooting guide](https://cloud.google.com/run/docs/troubleshooting)

---

## Live Deployment

**Current deployment**: https://deberta-anli-nli-66725207998.us-east1.run.app

- Model: DeBERTa-v3-base + LoRA (r=16)
- Dataset: ANLI R2
- Accuracy: 50.3%
- Response time: ~2-3 seconds
