from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ANLI NLI Inference API",
    description="Natural Language Inference using fine-tuned DeBERTa-v3-base on ANLI R2",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = None
tokenizer = None
device = None

class PredictionRequest(BaseModel):
    premise: str
    hypothesis: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "premise": "A person is walking a dog in the park",
                "hypothesis": "A person is outside"
            }
        }

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    
class BatchPredictionRequest(BaseModel):
    pairs: List[Dict[str, str]]
    
    class Config:
        json_schema_extra = {
            "example": {
                "pairs": [
                    {
                        "premise": "A person is walking a dog",
                        "hypothesis": "A person is outside"
                    },
                    {
                        "premise": "The cat is sleeping",
                        "hypothesis": "The cat is awake"
                    }
                ]
            }
        }

@app.on_event("startup")
async def load_model():
    """Load model and tokenizer from local directory (downloaded during build)"""
    global model, tokenizer, device
    
    logger.info("Loading model from local directory...")
    
    try:
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        

        model_dir = "./model"
        
        logger.info(f"Loading from: {model_dir}")
        
     
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=3)
        model.to(device)
        model.eval()
        
        logger.info("Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "ANLI NLI Inference API",
        "version": "1.0.0",
        "model": "DeBERTa-v3-base fine-tuned on ANLI R2",
        "accuracy": "50.3%",
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "health": "/health"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Predict the relationship between premise and hypothesis
    
    Returns:
    - prediction: Entailment, Neutral, or Contradiction
    - confidence: Probability of the predicted class
    - probabilities: Probabilities for all classes
    """
    try:
        # Tokenize input
        inputs = tokenizer(
            request.premise,
            request.hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True
        ).to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]
            pred_idx = torch.argmax(probs).item()
        
        # Map to labels
        labels = ["Entailment", "Neutral", "Contradiction"]
        
        return PredictionResponse(
            prediction=labels[pred_idx],
            confidence=float(probs[pred_idx]),
            probabilities={
                "entailment": float(probs[0]),
                "neutral": float(probs[1]),
                "contradiction": float(probs[2])
            }
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict")
def batch_predict(request: BatchPredictionRequest):
    """
    Predict relationships for multiple premise-hypothesis pairs
    
    Returns list of predictions
    """
    try:
        results = []
        
        for pair in request.pairs:
            # Tokenize
            inputs = tokenizer(
                pair["premise"],
                pair["hypothesis"],
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True
            ).to(device)
            
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)[0]
                pred_idx = torch.argmax(probs).item()
            
            labels = ["Entailment", "Neutral", "Contradiction"]
            
            results.append({
                "premise": pair["premise"],
                "hypothesis": pair["hypothesis"],
                "prediction": labels[pred_idx],
                "confidence": float(probs[pred_idx]),
                "probabilities": {
                    "entailment": float(probs[0]),
                    "neutral": float(probs[1]),
                    "contradiction": float(probs[2])
                }
            })
        
        return {"results": results}
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
