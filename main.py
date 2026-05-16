import sys
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from medhack_ai_assistant.config import ID_COLUMN
from medhack_ai_assistant.pipeline import analyze_patient_exam, run_quality_gate
from medhack_ai_assistant.services.dashboard import build_dashboard, load_backend_dataset


app = FastAPI(
    title="MedHack AI Assistant API",
    version="0.1.0",
    description="Runtime API for occupational health contraindication analysis.",
)


class AnalyzeRequest(BaseModel):
    exam_row_id: int | None = Field(
        default=None,
        description="Exam row id to load from train/test dataset.",
    )
    use_train: bool = Field(
        default=False,
        description="Read exam_row_id from train.csv instead of test.csv.",
    )
    payload: dict[str, Any] | None = Field(
        default=None,
        description="Full patient package. Takes precedence over exam_row_id.",
    )


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/api/v1/analyze")
async def analyze(request: AnalyzeRequest) -> dict[str, Any]:
    """Analyze patient for contraindications."""
    row = _resolve_patient_row(request)
    result = await analyze_patient_exam(row)
    return jsonable_encoder(result.to_dict())


@app.post("/api/v1/quality-gate")
async def quality_gate(request: AnalyzeRequest) -> dict[str, Any]:
    """Check if patient data passes quality gate."""
    row = _resolve_patient_row(request)
    result = run_quality_gate(row)
    return jsonable_encoder(result)


@app.get("/api/v1/dashboard/{exam_row_id}")
async def dashboard(exam_row_id: int, use_train: bool = False) -> dict[str, Any]:
    """Get dashboard data for a patient."""
    row = _load_exam_row(exam_row_id, use_train=use_train)
    return jsonable_encoder(build_dashboard(row).to_dict())


def _resolve_patient_row(request: AnalyzeRequest) -> pd.Series | dict[str, Any]:
    """Resolve patient row from request payload or ID."""
    if request.payload is not None:
        return request.payload

    if request.exam_row_id is None:
        raise HTTPException(
            status_code=422,
            detail="Provide either payload or exam_row_id.",
        )

    return _load_exam_row(request.exam_row_id, use_train=request.use_train)


def _load_exam_row(exam_row_id: int, *, use_train: bool) -> pd.Series:
    """Load exam row from dataset by ID."""
    dataset = load_backend_dataset(use_train=use_train)
    matched = dataset.loc[dataset[ID_COLUMN] == exam_row_id]

    if matched.empty:
        dataset_name = "train.csv" if use_train else "test.csv"
        raise HTTPException(
            status_code=404,
            detail=f"Exam row {exam_row_id} was not found in {dataset_name}.",
        )

    return matched.iloc[0]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
