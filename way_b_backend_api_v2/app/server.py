"""FastAPI Microservice for Laya Address VN v2 (Way B — CPU Backend, port 8084).

Launch with:
    .venv\\Scripts\\python.exe -m app.server
or:
    .venv\\Scripts\\uvicorn.exe app.server:app --host 127.0.0.1 --port 8084
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

from app.engine import InferenceMode, LayaAddressEngine, SufficiencyLabel

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML_PATH = STATIC_DIR / "index.html"

_engine: Optional[LayaAddressEngine] = None


def get_engine() -> LayaAddressEngine:
    global _engine
    if _engine is None:
        _engine = LayaAddressEngine()
    return _engine


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_engine()
    yield


app = FastAPI(
    title="Laya for Address VN v2 — Way B CPU Microservice",
    description=(
        "Non-autoregressive Dual-Granularity Transformer + Char-CNN 3-Class Sufficiency API "
        "powered by calibrated ONNX Runtime INT8 + MIT Vietnam Admin Gazetteer Tier-1 heuristics."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


class DecideRequest(BaseModel):
    text: str = Field(..., max_length=1000, description="Vietnamese input text to classify.")
    mode: InferenceMode = Field(
        default="hybrid",
        description="Routing mode: 'hybrid' | 'model_only' | 'heuristic_only'.",
    )


class DecideResponse(BaseModel):
    text: str
    is_address: bool
    decision: Literal["yes", "no"]
    probability: float
    confidence: float
    sufficiency: SufficiencyLabel
    sufficiency_probs: Dict[str, float]
    sufficiency_score: float
    tier: str
    reason: str
    latency_ms: float
    mode: InferenceMode


class BatchDecideRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of Vietnamese strings to classify in a vectorized batch.",
    )
    mode: InferenceMode = Field(
        default="hybrid",
        description="Routing mode: 'hybrid' | 'model_only' | 'heuristic_only'.",
    )


class BatchDecideResponse(BaseModel):
    mode: InferenceMode
    count: int
    total_latency_ms: float
    avg_latency_ms: float
    throughput_req_per_s: float
    tier_counts: Dict[str, int]
    sufficiency_counts: Dict[str, int]
    results: List[Dict[str, Any]]


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_web_ui() -> HTMLResponse:
    if not INDEX_HTML_PATH.exists():
        raise HTTPException(status_code=404, detail="Web Test UI not found.")
    return HTMLResponse(content=INDEX_HTML_PATH.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> Dict[str, Any]:
    eng = get_engine()
    return {
        "status": "ok",
        "version": "2.0.0",
        "model_loaded": True,
        "model_file": eng.model_path.name,
        "temperature": eng.temperature,
        "sufficiency_temperature": eng.suf_temperature,
        "provider": "CPUExecutionProvider",
    }


@app.get("/v1/info")
def info() -> Dict[str, Any]:
    return get_engine().get_info()


@app.post("/v1/decide", response_model=DecideResponse)
def decide(payload: DecideRequest) -> DecideResponse:
    eng = get_engine()
    res = eng.decide(payload.text, mode=payload.mode)
    return DecideResponse(**res.to_dict(), mode=payload.mode)


@app.post("/v1/decide/batch", response_model=BatchDecideResponse)
def decide_batch(payload: BatchDecideRequest) -> BatchDecideResponse:
    eng = get_engine()
    batch_out = eng.decide_batch(payload.texts, mode=payload.mode)
    return BatchDecideResponse(**batch_out)


if __name__ == "__main__":
    uvicorn.run("app.server:app", host="127.0.0.1", port=8084, reload=False)
