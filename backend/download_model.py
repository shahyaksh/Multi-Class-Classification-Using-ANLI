"""
Script to download W&B model artifact and base model during Docker build
"""
import wandb
import os
import sys
from huggingface_hub import login as hf_login
from transformers import AutoModelForSequenceClassification, AutoTokenizer

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "capcool79-northeastern-university")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "EAI-Assignment-Multi-class-classification-using-ANLI")
WANDB_ARTIFACT = os.getenv("WANDB_ARTIFACT", "model-deberta-base-lora-r16-r2:v0")

if not WANDB_API_KEY:
    print("ERROR: WANDB_API_KEY environment variable not set")
    sys.exit(1)


print("Downloading base model from Hugging Face...")

if HF_TOKEN:
    print("Logging in to Hugging Face...")
    hf_login(token=HF_TOKEN)
else:
    print("WARNING: No HF_TOKEN provided, may hit rate limits")

BASE_MODEL = "microsoft/deberta-v3-base"
BASE_MODEL_DIR = "./base_model"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.save_pretrained(BASE_MODEL_DIR)

base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=3)
base_model.save_pretrained(BASE_MODEL_DIR)

print(f"Base model saved to: {BASE_MODEL_DIR}")


print(f"Downloading LoRA adapter from W&B...")

wandb.login(key=WANDB_API_KEY)

run = wandb.init(
    project=WANDB_PROJECT,
    entity=WANDB_ENTITY,
    job_type="model_download"
)

artifact_path = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{WANDB_ARTIFACT}"
artifact = run.use_artifact(artifact_path, type='model')
adapter_dir = artifact.download(root="./lora_adapter")

print(f"LoRA adapter downloaded to: {adapter_dir}")

wandb.finish()
print("Download complete!")
