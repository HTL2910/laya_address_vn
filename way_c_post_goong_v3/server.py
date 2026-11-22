"""FastAPI HTTP server for Post-Goong V3/V3.1 address fill verification on port 8085."""

from __future__ import annotations

from contextlib import asynccontextmanager
import os
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

try:
    from .verifier import PostGoongVerifier
except ImportError:  # pragma: no cover - direct script execution fallback
    from verifier import PostGoongVerifier

_VERIFIER: PostGoongVerifier | None = None


def get_verifier() -> PostGoongVerifier:
    """Return cached singleton PostGoongVerifier (reads LAYA_POST_GOONG_ARTIFACTS_DIR if set)."""
    global _VERIFIER
    if _VERIFIER is None:
        custom_dir = os.environ.get("LAYA_POST_GOONG_ARTIFACTS_DIR", "").strip()
        _VERIFIER = PostGoongVerifier(artifacts_dir=custom_dir) if custom_dir else PostGoongVerifier()
    return _VERIFIER


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_verifier()
    yield


app = FastAPI(
    title="Post-Goong V3.1 Address Verifier",
    version="3.1.0",
    lifespan=lifespan,
)


class VerifyRequest(BaseModel):
    raw_text: str = Field(..., description="Raw Vietnamese user address text.")
    goong_ward: str = Field(default="", description="Ward filled by Goong V1/V2.")
    goong_district: str = Field(default="", description="District filled by Goong (empty in V2).")
    goong_province: str = Field(default="", description="Province/city filled by Goong.")
    goong_score: Optional[float] = Field(default=None, description="Optional Goong match confidence.")
    goong_lat: Optional[float] = Field(default=None, description="Optional Goong latitude.")
    goong_lng: Optional[float] = Field(default=None, description="Optional Goong longitude.")


class VerifyResponse(BaseModel):
    action: str
    is_valid_fill: bool
    risk_score: float
    sufficiency_hint: str
    reason_code: str
    is_legacy_2025_equiv: bool
    spatial_dist_km: float
    zns_confirm_message: Optional[str] = None
    latency_ms: float


class VerifyBatchRequest(BaseModel):
    items: list[VerifyRequest]


class VerifyBatchResponse(BaseModel):
    results: list[VerifyResponse]
    count: int


@app.get("/health")
def health() -> dict[str, object]:
    v = get_verifier()
    return {
        "status": "ok",
        "version": "3.1.0",
        "feature_schema": v.feature_schema,
        "allow_missing_ward": v.allow_missing_ward,
        "provider": "CPUExecutionProvider",
        "t_low": v.t_low,
        "t_high": v.t_high,
        "temperature": v.temperature,
    }


@app.post("/v1/verify", response_model=VerifyResponse)
def verify_single(req: VerifyRequest) -> VerifyResponse:
    v = get_verifier()
    res = v.verify(
        raw_text=req.raw_text,
        goong_ward=req.goong_ward,
        goong_district=req.goong_district,
        goong_province=req.goong_province,
        goong_score=req.goong_score,
        goong_lat=req.goong_lat,
        goong_lng=req.goong_lng,
    )
    return VerifyResponse(**res)


@app.post("/v1/verify/batch", response_model=VerifyBatchResponse)
def verify_batch(req: VerifyBatchRequest) -> VerifyBatchResponse:
    v = get_verifier()
    items = [item.model_dump() for item in req.items]
    results = [VerifyResponse(**r) for r in v.verify_batch(items)]
    return VerifyBatchResponse(results=results, count=len(results))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8085)
