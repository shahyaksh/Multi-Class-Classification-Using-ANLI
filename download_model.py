"""
Script to download W&B model artifact during Docker build
"""
import wandb
import os
import sys

# W&B configuration from environment or defaults
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "capcool79-northeastern-university")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "EAI-Assignment-Multi-class-classification-using-ANLI")
WANDB_ARTIFACT = os.getenv("WANDB_ARTIFACT", "model-deberta-base-lora-r16-r2:v0")

if not WANDB_API_KEY:
    print("ERROR: WANDB_API_KEY environment variable not set")
    sys.exit(1)

print(f"Downloading model from W&B...")
print(f"Entity: {WANDB_ENTITY}")
print(f"Project: {WANDB_PROJECT}")
print(f"Artifact: {WANDB_ARTIFACT}")

# Login to W&B
wandb.login(key=WANDB_API_KEY)

# Initialize W&B run
run = wandb.init(
    project=WANDB_PROJECT,
    entity=WANDB_ENTITY,
    job_type="model_download"
)

# Download artifact
artifact_path = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{WANDB_ARTIFACT}"
print(f"Downloading: {artifact_path}")

artifact = run.use_artifact(artifact_path, type='model')
model_dir = artifact.download(root="./model")

print(f"Model downloaded to: {model_dir}")

# Finish W&B run
wandb.finish()

print("Download complete!")
