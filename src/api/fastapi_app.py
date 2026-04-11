"""
fastapi_app.py
==============
Production REST API for brain tumor MRI classification.

Endpoints:
  POST /predict          → classify + segment + XAI
  POST /predict/classify → classify only (fast)
  GET  /health           → health check
  GET  /models           → model metadata
  GET  /docs             → Swagger UI (auto-generated)

Run: uvicorn src.api.fastapi_app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image
from pydantic import BaseModel

from src.api.inference import BrainTumorInference

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Brain Tumor MRI Classifier API",
    description=(
        "Production-ready deep learning API for brain tumor detection and classification. "
        "Ensemble of EfficientNetV2-M + ResNet50-CBAM + ViT-B/16 with Grad-CAM++ XAI."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global inference engine (loaded once on startup)
# ---------------------------------------------------------------------------

_engine: Optional[BrainTumorInference] = None


@app.on_event("startup")
async def startup_event() -> None:
    global _engine
    config_path = os.environ.get("CONFIG_PATH", "config.yaml")
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "checkpoints")

    if not Path(config_path).exists():
        logger.warning(f"Config not found at {config_path} — API running without model.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading inference engine on {device}…")

    try:
        _engine = BrainTumorInference.from_config(config_path, checkpoint_dir, device)
        logger.info("Inference engine loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load inference engine: {e}")


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class ProbabilityBreakdown(BaseModel):
    meningioma: float
    glioma: float
    pituitary: float


class PredictionResult(BaseModel):
    class_index: int
    class_name: str
    confidence: float
    probabilities: ProbabilityBreakdown


class LatencyInfo(BaseModel):
    preprocessing: float
    classification: float
    segmentation: float
    xai: float
    total: float


class PredictResponse(BaseModel):
    sample_id: str
    prediction: PredictionResult
    latency_ms: LatencyInfo
    gradcam_paths: Optional[Dict[str, str]] = None
    segmentation_available: bool = False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health_check() -> Dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": _engine is not None,
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "cuda_available": torch.cuda.is_available(),
    }


@app.get("/models", tags=["System"])
async def model_info() -> Dict:
    """Return ensemble model metadata."""
    return {
        "ensemble": {
            "models": ["EfficientNetV2-M", "ResNet50-CBAM", "ViT-B/16"],
            "voting": "soft-voting",
            "temperature_scaling": True,
            "weights": {"efficientnet": 0.4, "resnet_cbam": 0.3, "vit": 0.3},
        },
        "segmentation": "U-Net with EfficientNet-B4 encoder",
        "xai": ["Grad-CAM++", "SHAP DeepExplainer", "Integrated Gradients"],
        "classes": ["meningioma", "glioma", "pituitary"],
        "target_metrics": {
            "accuracy": "≥99.1%",
            "auc": "≥0.995",
            "sensitivity": "≥96%",
            "specificity": "≥99%",
        },
    }


@app.post("/predict", tags=["Inference"])
async def predict(
    file: UploadFile = File(..., description="MRI image (PNG/JPG/DICOM)"),
    include_xai: bool = True,
    include_seg: bool = True,
) -> JSONResponse:
    """
    Full prediction pipeline: classify + segment + XAI heatmap.

    Accepts PNG / JPEG MRI images. Returns class prediction, confidence,
    segmentation mask info, and Grad-CAM++ heatmap paths.
    """
    if _engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Check server logs.",
        )

    # Read and decode image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Cannot decode image: {e}")

    sample_id = Path(file.filename).stem if file.filename else "upload"

    try:
        result = _engine.predict(
            image=image_np,
            sample_id=sample_id,
            include_xai=include_xai,
            include_seg=include_seg,
        )
    except Exception as e:
        logger.exception(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return JSONResponse(content=result)


@app.post("/predict/classify", tags=["Inference"])
async def classify_only(
    file: UploadFile = File(...),
) -> JSONResponse:
    """Fast classification only — no XAI or segmentation."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    contents = await file.read()
    try:
        image = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    result = _engine.predict(image, include_xai=False, include_seg=False)
    return JSONResponse(content={
        "prediction": result["prediction"],
        "latency_ms": result["latency_ms"],
    })


@app.get("/heatmap/{filename}", tags=["XAI"])
async def get_heatmap(filename: str) -> FileResponse:
    """Serve a previously generated Grad-CAM++ heatmap image."""
    heatmap_path = Path("outputs/heatmaps") / filename
    if not heatmap_path.exists():
        raise HTTPException(status_code=404, detail="Heatmap not found.")
    return FileResponse(str(heatmap_path), media_type="image/png")
