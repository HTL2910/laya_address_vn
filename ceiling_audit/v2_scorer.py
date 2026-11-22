"""In-process & Node+ORT live scorer for existing Laya Address VN v2 gates.

Used when the user's DEV CSV does not already include `way_a_v2_pass` / `way_b_v2_pass`
columns and `--score-v2` is passed to `ceiling_eval.py`. Runs 100% offline.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence
import unicodedata

import numpy as np
import onnxruntime as ort

LAYA_ROOT = Path(__file__).resolve().parent.parent
WAY_A_V2_DIR = LAYA_ROOT / "way_a_browser_onnx_v2"
WAY_B_V2_DIR = LAYA_ROOT / "way_b_backend_api_v2"


def score_way_b_v2_batch(
    texts: Sequence[str],
    mode: str = "hybrid",
    chunk_size: int = 128,
) -> Dict[str, List[Any]]:
    """Score texts using Way B v2 (`LayaAddressEngine` INT8 + Gazetteer)."""
    wb_str = str(WAY_B_V2_DIR)
    if wb_str not in sys.path:
        sys.path.insert(0, wb_str)

    from app.engine import LayaAddressEngine  # type: ignore

    engine = LayaAddressEngine(artifacts_dir=WAY_B_V2_DIR / "artifacts")
    res = engine.decide_batch(list(texts), mode=mode, chunk_size=chunk_size)["results"]
    return {
        "pass": [bool(r["is_address"]) for r in res],
        "probability": [float(r["probability"]) for r in res],
        "sufficiency": [str(r["sufficiency"]) for r in res],
    }


class WayAV2ChunkedScorer:
    """Chunked Way A v2 scorer (Node `web/heuristics.js` + ONNX Runtime Char-CNN)."""

    def __init__(self, way_a_dir: Path = WAY_A_V2_DIR) -> None:
        self.way_a_dir = way_a_dir
        self.node_bin = shutil.which("node")
        model_dir = way_a_dir / "web" / "model"
        vocab = json.loads((model_dir / "vocab.json").read_text(encoding="utf-8"))
        calib = json.loads((model_dir / "calibration.json").read_text(encoding="utf-8"))
        self.char_to_id: Dict[str, int] = vocab["char_to_id"]
        self.unk_id = int(vocab["unk_id"])
        self.max_len = int(vocab["max_length"])
        self.temp = float(calib["temperature"])
        self.threshold = float(calib["threshold"])
        self.t_suf = float(calib["sufficiency_temperature"])
        self.classes: List[str] = list(calib["sufficiency_classes"])

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(
            str(model_dir / "laya_address_vn.onnx"),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        # Warmup
        self._run_onnx_batch(["Lộc Thạnh, Lộc Ninh"])

    def _encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        arr = np.zeros((len(texts), self.max_len), dtype=np.int64)
        for r, text in enumerate(texts):
            norm = re.sub(r"\s+", " ", unicodedata.normalize("NFC", text or "").strip()).lower()
            for c, ch in enumerate(norm[: self.max_len]):
                arr[r, c] = self.char_to_id.get(ch, self.unk_id)
        return arr

    def _run_onnx_batch(self, texts: Sequence[str], batch_size: int = 256) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            x = self._encode_batch(chunk)
            logit_arr, suf_arr = self.sess.run(["logit", "sufficiency_logits"], {"input_ids": x})
            logits = np.asarray(logit_arr, dtype=np.float64).reshape(-1)
            sufs = np.asarray(suf_arr, dtype=np.float64)
            for logit, suf_raw in zip(logits, sufs):
                prob = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, float(logit) / self.temp))))
                scaled = suf_raw / self.t_suf
                scaled = scaled - np.max(scaled)
                exps = np.exp(np.clip(scaled, -30.0, 30.0))
                probs = (exps / np.sum(exps)).tolist()
                best_idx = int(np.argmax(probs))
                out.append({
                    "is_address": bool(prob >= self.threshold),
                    "probability": float(prob),
                    "sufficiency": self.classes[best_idx],
                })
        return out

    def _run_node_tier1_chunk(self, texts: Sequence[str]) -> List[Dict[str, Any]]:
        if not self.node_bin:
            raise RuntimeError("Node.js binary not found on PATH; cannot run Way A v2 heuristics.js")
        heur_js = str(self.way_a_dir / "web" / "heuristics.js").replace("\\", "\\\\")
        gaz_json = str(self.way_a_dir / "web" / "model" / "vn_admin_gazetteer.json").replace("\\", "\\\\")
        script = (
            "const fs = require('fs');"
            f"const H = require('{heur_js}');"
            f"H.initGazetteer(JSON.parse(fs.readFileSync('{gaz_json}', 'utf8')));"
            "let buf = '';"
            "process.stdin.on('data', d => buf += d);"
            "process.stdin.on('end', () => {"
            "  const arr = JSON.parse(buf);"
            "  const out = arr.map(t => H.evaluateTier1(t));"
            "  process.stdout.write(JSON.stringify(out));"
            "});"
        )
        proc = subprocess.run(
            [self.node_bin, "-e", script],
            input=json.dumps(list(texts), ensure_ascii=False),
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=True,
        )
        return json.loads(proc.stdout)

    def score_batch(
        self,
        texts: Sequence[str],
        mode: str = "hybrid",
        node_chunk_size: int = 4000,
    ) -> Dict[str, List[Any]]:
        n = len(texts)
        if n == 0:
            return {"pass": [], "probability": [], "sufficiency": []}

        if mode == "model_only":
            res = self._run_onnx_batch(texts)
            return {
                "pass": [bool(r["is_address"]) for r in res],
                "probability": [float(r["probability"]) for r in res],
                "sufficiency": [str(r["sufficiency"]) for r in res],
            }

        # Hybrid mode: chunked Node Tier-1 + batched ORT on ambiguous items
        t1_all: List[Dict[str, Any]] = []
        for start in range(0, n, node_chunk_size):
            t1_all.extend(self._run_node_tier1_chunk(texts[start : start + node_chunk_size]))

        passes = [False] * n
        probs = [0.0] * n
        sufs = ["not_address"] * n
        ambig_indices: List[int] = []
        ambig_texts: List[str] = []

        for idx, (txt, r) in enumerate(zip(texts, t1_all)):
            if r["decision"] != "ambiguous":
                passes[idx] = bool(r["is_address"])
                probs[idx] = float(r["probability"])
                sufs[idx] = str(r["sufficiency"])
            else:
                ambig_indices.append(idx)
                ambig_texts.append(txt)

        if ambig_texts:
            t2_res = self._run_onnx_batch(ambig_texts)
            for idx, r in zip(ambig_indices, t2_res):
                passes[idx] = bool(r["is_address"])
                probs[idx] = float(r["probability"])
                sufs[idx] = str(r["sufficiency"])

        return {"pass": passes, "probability": probs, "sufficiency": sufs}


def score_way_a_v2_batch(
    texts: Sequence[str],
    mode: str = "hybrid",
    node_chunk_size: int = 4000,
) -> Optional[Dict[str, List[Any]]]:
    """Score texts using Way A v2 if Node.js is available; returns None with warning if absent."""
    if not shutil.which("node"):
        print("[WARN] Node.js không có trong PATH — bỏ qua chấm tự động Way A v2 (chỉ chạy Way B v2).", file=sys.stderr)
        return None
    scorer = WayAV2ChunkedScorer(WAY_A_V2_DIR)
    return scorer.score_batch(texts, mode=mode, node_chunk_size=node_chunk_size)
