from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(
    title="Disease Prediction API",
    version="1.0.0",
    description="ML-based disease risk prediction system"
)

models = {}

class PredictionRequest(BaseModel):
    disease_type: str
    features: List[float]

class PredictionResponse(BaseModel):
    disease_type: str
    prediction: int
    probability: float
    risk_level: str
    message: str

@app.on_event("startup")
async def load_models():
    global models
    
    diseases = ['diabetes', 'heart', 'liver', 'kidney']
    
    for disease in diseases:
        try:
            model_path = f'models/{disease}_ensemble.pkl'
            if os.path.exists(model_path):
                models[disease] = joblib.load(model_path)
        except Exception as e:
            print(f"Error loading {disease}: {e}")

@app.get("/")
async def root():
    return {
        "status": "Disease Prediction API Active",
        "version": "1.0.0",
        "supported_diseases": list(models.keys()),
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "Healthy",
        "models_loaded": list(models.keys()),
        "ready": len(models) > 0
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if request.disease_type not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Disease '{request.disease_type}' not supported"
        )
    
    if len(request.features) == 0:
        raise HTTPException(status_code=400, detail="Features cannot be empty")
    
    try:
        model = models[request.disease_type]
        X = np.array(request.features).reshape(1, -1)
        
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]
        
        if probability > 0.7:
            risk_level = "High Risk"
            message = "Immediate medical attention recommended"
        elif probability > 0.4:
            risk_level = "Medium Risk"
            message = "Consult with healthcare professional"
        else:
            risk_level = "Low Risk"
            message = "Continue regular check-ups"
        
        return PredictionResponse(
            disease_type=request.disease_type,
            prediction=int(prediction),
            probability=float(probability),
            risk_level=risk_level,
            message=message
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict-batch")
async def predict_batch(requests: List[PredictionRequest]):
    results = []
    for req in requests:
        result = await predict(req)
        results.append(result)
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)