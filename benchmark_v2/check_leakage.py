"""Independent N-Gram & Exact Leakage Auditor for Laya Address VN v2.

Checks training manifests of Way A v2 and Way B v2 against:
- benchmark/vn_address_benchmark.json (90 cases)
- benchmark_v2/independent_holdout_v2.json (600 cases)
Also documents the 5 leaked cases found in Way A v1 dataset.py (C01, NN02, NN03, NN09, NN10).
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
OLD_BENCH = ROOT / "benchmark" / "vn_address_benchmark.json"
NEW_BENCH = ROOT / "benchmark_v2" / "independent_holdout_v2.json"
WAY_A_V1_DS = ROOT / "way_a_browser_onnx" / "train" / "dataset.py"
WAY_A_V2_MANIFEST = ROOT / "way_a_browser_onnx_v2" / "train" / "train_texts_manifest.txt"
WAY_B_V2_MANIFEST = ROOT / "way_b_backend_api_v2" / "train" / "train_texts_manifest.txt"
OUT_REPORT = ROOT / "benchmark_v2" / "leakage_report.json"


def strip_accents(s: str) -> str:
    s = (s or "").replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", strip_accents(s).lower().strip())


def norm_raw(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s or "").strip().lower())


def word_ngrams(s: str, n: int = 4) -> Set[Tuple[str, ...]]:
    toks = re.findall(r"[a-z0-9]+", norm_text(s))
    if len(toks) < n:
        return set()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def char_ngrams(s: str, n: int = 15) -> Set[str]:
    t = norm_text(s)
    if len(t) < n:
        return set()
    return {t[i : i + n] for i in range(len(t) - n + 1)}


def audit_manifest(manifest_path: Path, eval_sets: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    raw_lines = [line.rstrip("\n") for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    train_raw = {norm_raw(x) for x in raw_lines}
    train_norm = {norm_text(x) for x in raw_lines}

    train_w4: Set[Tuple[str, ...]] = set()
    train_c15: Set[str] = set()
    for line in raw_lines:
        train_w4.update(word_ngrams(line, 4))
        train_c15.update(char_ngrams(line, 15))

    report: Dict[str, Any] = {"train_manifest_lines": len(raw_lines), "datasets": {}}
    for ds_name, cases in eval_sets.items():
        exact_raw_hits = []
        exact_norm_hits = []
        w4_overlap_cases = 0
        c15_overlap_cases = 0

        for c in cases:
            txt = c["text"]
            if norm_raw(txt) in train_raw:
                exact_raw_hits.append({"id": c["id"], "text": txt})
            if norm_text(txt) in train_norm:
                exact_norm_hits.append({"id": c["id"], "text": txt})
            w4 = word_ngrams(txt, 4)
            if w4 and (w4 & train_w4):
                w4_overlap_cases += 1
            c15 = char_ngrams(txt, 15)
            if c15 and (c15 & train_c15):
                c15_overlap_cases += 1

        report["datasets"][ds_name] = {
            "eval_cases": len(cases),
            "exact_raw_matches": len(exact_raw_hits),
            "exact_norm_matches": len(exact_norm_hits),
            "exact_hits": exact_norm_hits,
            "word_4gram_overlap_cases": w4_overlap_cases,
            "word_4gram_overlap_rate": round(w4_overlap_cases / max(1, len(cases)), 4),
            "char_15gram_overlap_cases": c15_overlap_cases,
            "char_15gram_overlap_rate": round(c15_overlap_cases / max(1, len(cases)), 4),
        }
    return report


def audit_v1_way_a_known_leakage(old_cases: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    if not WAY_A_V1_DS.exists():
        return []
    src = WAY_A_V1_DS.read_text(encoding="utf-8")
    leaks = []
    for c in old_cases:
        if c["text"] in src:
            leaks.append({"id": c["id"], "text": c["text"]})
    return leaks


def main() -> int:
    old_cases = json.loads(OLD_BENCH.read_text(encoding="utf-8"))["cases"]
    new_cases = json.loads(NEW_BENCH.read_text(encoding="utf-8"))["cases"]
    eval_sets = {
        "old_90_benchmark": old_cases,
        "independent_600_holdout": new_cases,
    }

    result: Dict[str, Any] = {
        "v1_way_a_literal_source_leaks": audit_v1_way_a_known_leakage(old_cases),
    }

    if WAY_A_V2_MANIFEST.exists():
        result["way_a_v2"] = audit_manifest(WAY_A_V2_MANIFEST, eval_sets)
    if WAY_B_V2_MANIFEST.exists():
        result["way_b_v2"] = audit_manifest(WAY_B_V2_MANIFEST, eval_sets)

    OUT_REPORT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
