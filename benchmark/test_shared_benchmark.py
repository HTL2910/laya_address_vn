"""Pytest wrapper ensuring both Way A and Way B clear quality thresholds on the 90-case held-out benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from run_shared_eval import BENCHMARK_PATH, eval_way_a, eval_way_b


def test_shared_90_case_benchmark_way_a_and_way_b() -> None:
    cases = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))["cases"]
    assert len(cases) == 90

    res_a = eval_way_a(cases)
    res_b = eval_way_b(cases)

    for name, metrics in {**res_a, **res_b}.items():
        assert metrics["accuracy"] >= 0.85, (
            f"{name} accuracy {metrics['accuracy']:.2%} < 85%: {metrics['mistakes']}"
        )
