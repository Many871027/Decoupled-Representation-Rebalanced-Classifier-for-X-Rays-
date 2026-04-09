from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import mlflow
from mlflow.tracking import MlflowClient
import tensorflow as tf
from pathlib import Path
import numpy as np
from PIL import Image
import io

from src.config import CLASS_NAMES, MLFLOW_TRACKING_URI, EXPERIMENT_NAME, IMG_HEIGHT, IMG_WIDTH, CHANNELS

app = FastAPI(
    title="Chest X-Ray Senior Classifier API",
    description="API for Image classification and MLOps metrics retrieval.",
    version="1.0"
)

# Mocked or generic global model loaded later in real deployment
model = None 

# Initializer script triggered by Fast API runtime
@app.on_event("startup")
async def load_model():
    # In production, loading best model from MLFlow registry dynamically.
    # Currently omitted until experiment generates an artifact
    pass

class MLFlowMetricsResponse(BaseModel):
    run_id: str
    status: str
    metrics: dict
    params: dict

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """ Endpoint to process and classify unseen XRay images. """
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="El archivo proporcionado no es un formato de imagen soportado")
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    if CHANNELS == 1:
        image = image.convert('L') # Convert to grayscale
    else:
        image = image.convert('RGB')
        
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=-1 if CHANNELS==1 else 0) # Format structure
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    
    if model is None:
        # Mock Response if model not yet trained in workflow
        return {"status": "Model not yet trained or loaded.", "filename": file.filename}
        
    preds = model.predict(img_array)
    pred_class = np.argmax(preds[0])
    confidence = float(preds[0][pred_class])
    
    return {
        "filename": file.filename,
        "prediction": CLASS_NAMES[pred_class],
        "confidence": confidence
    }

@app.get("/metrics", response_model=list[MLFlowMetricsResponse])
async def get_mlflow_metrics(limit: int = 5):
    """
    Recupera los ultimos registros de experimentacion usando el cliente MLFlow.
    Usa Pydantic schema validation.
    """
    try:
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        
        if not experiment:
            return []
            
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=limit,
            order_by=["metrics.macro_f1 DESC"]
        )
        
        responses = []
        for run in runs:
            responses.append(MLFlowMetricsResponse(
                run_id=run.info.run_id,
                status=run.info.status,
                metrics=run.data.metrics,
                params=run.data.params
            ))
            
        return responses
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
