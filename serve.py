import os
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Prediction Service")

MODEL_PATH = os.environ.get("MODEL_PATH", "models/best_model.joblib")
LABEL_ENCODER_PATH = os.environ.get("LABEL_ENCODER_PATH", "models/label_encoder.joblib")

# Try multiple fallback model filenames
FALLBACK_MODELS = ["models/best_model.joblib", "models/model_xgb.joblib", "models/model_logreg.joblib", "model.joblib", "model_xgb.joblib"]

model = None
label_encoder = None

# Load model with fallback
for p in [MODEL_PATH] + FALLBACK_MODELS:
    if os.path.exists(p):
        try:
            model = joblib.load(p)
            break
        except Exception:
            continue

# Load label encoder if present
if os.path.exists(LABEL_ENCODER_PATH):
    try:
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
    except Exception:
        label_encoder = None

class PredictRequest(BaseModel):
    rows: List[Dict[str, Any]]
    return_proba: Optional[bool] = True

@app.get("/")
def root():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "model_path": MODEL_PATH}

@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="No model found on server")

    # Build DataFrame from input rows
    try:
        df = pd.DataFrame(req.rows)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input rows: {e}")

    # Run prediction
    try:
        preds = model.predict(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    # If predictions are encoded and label_encoder exists, decode
    if label_encoder is not None:
        try:
            preds_out = label_encoder.inverse_transform(preds)
        except Exception:
            preds_out = preds.tolist()
    else:
        preds_out = preds.tolist()

    result = {"predictions": preds_out}

    if req.return_proba and hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(df)
            # If label encoder exists, label columns by original labels
            if label_encoder is not None:
                cols = [f"prob_{c}" for c in label_encoder.classes_]
            else:
                # numeric class labels
                cols = [f"prob_{i}" for i in range(probs.shape[1])]
            probs_list = [dict(zip(cols, row.tolist())) for row in probs]
            result["probabilities"] = probs_list
        except Exception:
            result["probabilities"] = None

    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=True)
