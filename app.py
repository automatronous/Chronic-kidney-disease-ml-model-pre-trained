from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib

app = FastAPI(title="CKD Prediction API", version="1.0.0")

# Load model (add error handling)
try:
    model = joblib.load("ckd_model.joblib")
except FileNotFoundError:
    model = None

class CKDInput(BaseModel):
    blood_pressure: float = Field(..., ge=50, le=200)
    specific_gravity: float = Field(..., ge=1.000, le=1.040)
    albumin: float = Field(..., ge=0, le=5)
    sugar: float = Field(..., ge=0, le=5)
    red_blood_cells: float = Field(..., ge=0, le=1)
    blood_urea_nitrogen: float = Field(..., ge=5, le=200)
    serum_creatinine: float = Field(..., ge=0.1, le=20.0)
    sodium: float = Field(..., ge=120, le=160)
    potassium: float = Field(..., ge=2.0, le=8.0)
    hemoglobin: float = Field(..., ge=5.0, le=20.0)
    white_blood_cell_count: float = Field(..., ge=1000, le=30000)
    red_blood_cell_count: float = Field(..., ge=2.0, le=8.0)
    hypertension: float = Field(..., ge=0, le=1)

@app.get("/")
def home():
    return {"message": "CKD model API is running"}

@app.post("/predict")
def predict(data: CKDInput):
    if model is None:
        return {"error": "Model file ckd_model.joblib not found"}
    
    # Convert new field names to old model column names
    model_data = {
        'bp': data.blood_pressure,
        'sg': data.specific_gravity,
        'al': data.albumin,
        'su': data.sugar,
        'rbc': data.red_blood_cells,
        'bu': data.blood_urea_nitrogen,
        'sc': data.serum_creatinine,
        'sod': data.sodium,
        'pot': data.potassium,
        'hemo': data.hemoglobin,
        'wbcc': data.white_blood_cell_count,
        'rbcc': data.red_blood_cell_count,
        'htn': data.hypertension
    }
    
    input_data = pd.DataFrame([model_data])
    prediction = model.predict(input_data)[0]
    
    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(max(model.predict_proba(input_data)[0]))
    
    return {
        "prediction": int(prediction),
        "confidence": confidence,
        "note": "Educational use only. Not a medical diagnosis."
    }