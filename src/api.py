"""
FastAPI backend untuk deteksi dini risiko stroke (prediksi per pasien).

Endpoint:
    GET  /         → info & status model
    GET  /health   → health check
    GET  /schema   → feature schema (untuk frontend dinamis)
    POST /predict  → prediksi 1 pasien

Run:
    uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import joblib
import mlflow.sklearn
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# ---------------------------------------------------------------- load artifacts at startup
MODEL = mlflow.sklearn.load_model(str(MODELS_DIR / "best_model"))
THRESHOLD = float(joblib.load(MODELS_DIR / "best_model" / "threshold.joblib")["threshold"])
SCALER = joblib.load(DATA_PROCESSED / "scaler.joblib")
with open(DATA_PROCESSED / "encoders.json") as f:
    ENCODERS: dict[str, dict[str, int]] = json.load(f)
with open(DATA_PROCESSED / "tabnet_meta.json") as f:
    META = json.load(f)
with open(MODELS_DIR / "best_model.json") as f:
    BEST_META = json.load(f)

FEATURE_ORDER: list[str] = META["feature_names"]
NUMERIC_FEATURES: list[str] = META["numeric_features"]  # age, avg_glucose_level, bmi


# ---------------------------------------------------------------- pydantic schema
class PatientInput(BaseModel):
    """Input data pasien — semua field WAJIB diisi."""
    gender: Literal["Male", "Female"] = Field(..., description="Jenis kelamin")
    age: float = Field(..., ge=0, le=120, description="Usia (tahun)")
    hypertension: Literal[0, 1] = Field(..., description="0=tidak, 1=hipertensi")
    heart_disease: Literal[0, 1] = Field(..., description="0=tidak, 1=ada penyakit jantung")
    ever_married: Literal["Yes", "No"] = Field(..., description="Status pernikahan")
    work_type: Literal["Private", "Self-employed", "Govt_job", "children", "Never_worked"] = Field(
        ..., description="Jenis pekerjaan"
    )
    Residence_type: Literal["Urban", "Rural"] = Field(..., description="Tipe tempat tinggal")
    avg_glucose_level: float = Field(..., ge=0, le=400, description="Kadar glukosa rata-rata")
    bmi: float = Field(..., ge=10, le=80, description="Body Mass Index")
    smoking_status: Literal["formerly smoked", "never smoked", "smokes", "Unknown"] = Field(
        ..., description="Status merokok"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Male", "age": 67, "hypertension": 0, "heart_disease": 1,
                "ever_married": "Yes", "work_type": "Private", "Residence_type": "Urban",
                "avg_glucose_level": 228.69, "bmi": 36.6, "smoking_status": "formerly smoked",
            }
        }
    }


class PredictionOutput(BaseModel):
    probability: float = Field(..., description="Probabilitas stroke (0-1)")
    threshold: float = Field(..., description="Cut-off keputusan model")
    prediction: int = Field(..., description="0 = tidak stroke, 1 = stroke")
    risk_label: str = Field(..., description="Label risiko: Low / Moderate / High / Very High")
    model_name: str = Field(..., description="Nama model yang dipakai")


# ---------------------------------------------------------------- preprocessing
def _encode_row(p: PatientInput) -> np.ndarray:
    """Konversi raw input → vektor fitur sesuai urutan training."""
    raw = p.model_dump()
    row = []
    for feat in FEATURE_ORDER:
        val = raw[feat]
        if feat in ENCODERS:
            mapping = ENCODERS[feat]
            if val not in mapping:
                raise HTTPException(400, f"Nilai '{val}' untuk fitur '{feat}' tidak dikenali. "
                                         f"Pilihan valid: {list(mapping.keys())}")
            row.append(mapping[val])
        else:
            row.append(val)
    arr = np.array(row, dtype=float).reshape(1, -1)

    # Scale numeric features (in-place pada index yang sesuai)
    num_idx = [FEATURE_ORDER.index(n) for n in NUMERIC_FEATURES]
    arr[:, num_idx] = SCALER.transform(arr[:, num_idx])
    return arr


def _risk_label(prob: float) -> str:
    if prob < 0.25: return "Low"
    if prob < 0.50: return "Moderate"
    if prob < 0.75: return "High"
    return "Very High"


def _predict(patient: PatientInput) -> PredictionOutput:
    X = _encode_row(patient)
    prob = float(MODEL.predict_proba(X)[0, 1])
    return PredictionOutput(
        probability=prob,
        threshold=THRESHOLD,
        prediction=int(prob >= THRESHOLD),
        risk_label=_risk_label(prob),
        model_name=f"{BEST_META['run_name']} ({BEST_META['family']})",
    )


# ---------------------------------------------------------------- app
app = FastAPI(
    title="Stroke Risk Prediction API",
    description="API klasifikasi risiko stroke berdasarkan data klinis & gaya hidup. "
                "Model: Logistic Regression (tuned, threshold-optimized) dengan PR-AUC primary metric.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "name": "Stroke Risk Prediction API",
        "version": "1.0.0",
        "model": f"{BEST_META['run_name']} ({BEST_META['family']})",
        "threshold": THRESHOLD,
        "primary_metric": BEST_META["primary_metric"],
        "test_metrics": BEST_META["test_metrics"],
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.get("/schema")
def schema():
    """Return feature schema agar frontend bisa generate form dinamis."""
    return {
        "features": FEATURE_ORDER,
        "numeric": NUMERIC_FEATURES,
        "categorical": {k: list(v.keys()) for k, v in ENCODERS.items()},
        "binary": ["hypertension", "heart_disease"],
        "ranges": {
            "age": [0, 120],
            "avg_glucose_level": [0, 400],
            "bmi": [10, 80],
        },
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(patient: PatientInput):
    return _predict(patient)
