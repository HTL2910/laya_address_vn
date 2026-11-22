"""Acceptance Criteria Gate Tests for Laya Address VN v2 (Section 6 + Section 4)."""

from __future__ import annotations

import json
from pathlib import Path

from eval_all_v2 import REPORT_OUT, WAY_A_V2_DIR

LEAKAGE_PATH = Path(__file__).resolve().parent / "leakage_report.json"


def test_zero_exact_leakage_in_v2() -> None:
    leak = json.loads(LEAKAGE_PATH.read_text(encoding="utf-8"))
    for way in ("way_a_v2", "way_b_v2"):
        for ds_name, ds in leak[way]["datasets"].items():
            assert ds["exact_raw_matches"] == 0, f"{way} leaked raw matches on {ds_name}: {ds['exact_hits']}"
            assert ds["exact_norm_matches"] == 0, f"{way} leaked norm matches on {ds_name}: {ds['exact_hits']}"


def test_way_a_v2_deployment_constraints() -> None:
    model_dir = WAY_A_V2_DIR / "web" / "model"
    assert not (model_dir / "laya_address_vn.onnx.data").exists()
    assert not list(WAY_A_V2_DIR.rglob("*.pt"))
    total_bytes = sum(
        (model_dir / fname).stat().st_size
        for fname in ("laya_address_vn.onnx", "vn_admin_gazetteer.json", "vocab.json", "calibration.json")
    )
    assert total_bytes <= 2 * 1024 * 1024, f"Way A v2 artifacts exceed 2 MB: {total_bytes}"


def test_section_6_acceptance_criteria_hybrid() -> None:
    rep = json.loads(REPORT_OUT.read_text(encoding="utf-8"))
    for key in ("way_a_v2_hybrid", "way_b_v2_hybrid"):
        h = rep["results"][key]["holdout_600"]

        # 1. Garbage block >= 95% + all 11 mandatory strings blocked
        assert h["garbage_block_rate"] >= 0.95, f"{key} garbage_block_rate={h['garbage_block_rate']}"
        for m_txt, m_res in h["mandatory_garbage_checks"].items():
            assert m_res["blocked"], f"{key} failed to block mandatory string: {m_txt!r} ({m_res})"

        # 2. Short valid address false-block <= 5% across EVERY ONE of the 6 styles
        for style, st_metrics in h["short_styles"].items():
            assert st_metrics["false_block_rate"] <= 0.05, (
                f"{key} false-blocked > 5% on short style {style}: {st_metrics}"
            )

        # 3. Insufficient address recall >= 90%
        assert h["insufficient_tagged_recall"] >= 0.90, (
            f"{key} insufficient_tagged_recall={h['insufficient_tagged_recall']}"
        )

        # 4. CPU p99 <= 10 ms
        assert h["latency_ms"]["p99"] <= 10.0, f"{key} p99={h['latency_ms']['p99']}"
