# Model Card: PostGoongVerifierMLP V3 (`way_c_post_goong_v3`)

## Overview
`PostGoongVerifierMLP` V3 is a self-contained, zero-PyTorch, zero-pickle calibrated binary risk classifier exported to single-file **ONNX opset 14** (`artifacts/post_goong_verifier_v3.onnx`) for verifying Goong V1 (3-tier 63-province legacy) and Goong V2 (2-tier 34-province 2025) Vietnamese address fills.

- **Architecture:** `MLP [43 -> 128 -> 64 -> 32 -> 1]` (`BatchNorm1d`, `SiLU`, `Dropout(0.20)`)
- **Parameters:** `16,449` (budget `<= 150,000`)
- **ONNX Artifact:** `artifacts/post_goong_verifier_v3.onnx` (`68.53 KB`, opset `14`, single-file, max PyTorch parity diff `1.91e-06`)
- **Runtime Dependencies:** Pure NumPy + `onnxruntime` (`CPUExecutionProvider`, `intra_op_num_threads=1`) — zero PyTorch, zero pickle, zero `.pt` files in runtime paths.
- **Serving API:** FastAPI service on port `8085` (`GET /health`, `POST /v1/verify`, `POST /v1/verify/batch`).

## Domain-Invariant Feature Schema (`43` features, Zero Admin IDs)
Extracted by `GazetteerV3.extract(...)` in `features.py` using bundled `artifacts/vn_admin_v3_converter.json`:
1. **Pre-Goong Structure (17):** `char_len`, `word_count`, `digit_count`, `comma_count`, `has_house_num`, `has_ward_pfx`, `has_dist_pfx`, `has_prov_pfx`, `has_street_kw`, `has_rural_kw`, `has_apt_kw`, `has_chat_food_cue`, `raw_ward_hits`, `raw_dist_hits`, `raw_prov_hits`, `raw_valid_pair`, `raw_min_ward_ambig`.
2. **Post-Goong Grounding (11):** `goong_has_ward`, `goong_has_dist`, `goong_has_prov`, `goong_filled_levels`, `ward_exact_grounded`, `dist_exact_grounded`, `prov_exact_grounded`, `ward_char_sim`, `dist_char_sim`, `prov_char_sim`, `admin_token_grounding_ratio`.
3. **Post-Goong Error Signals (10):** `hallucinated_levels_count`, `ward_hallucinated`, `parent_hallucinated`, `ambiguous_ward_ungrounded_parent`, `raw_prov_contradicts_goong`, `raw_dist_contradicts_goong`, `goong_ward_prov_count`, `log_goong_ward_ambiguity`, `goong_score`, `goong_score_missing`.
4. **2025 Equivalence & Spatial Checks (5):** `is_legacy_2025_equiv` (forward `eo`/`isle` and reverse `rev`), `merged_prov_equiv` (`pm[pid_o] == pid_n`), `spatial_dist_km` (Haversine against `ogeo`/`en`), `spatial_outlier` (`dist_km > max(20.0, 2.5 * sqrt(area) + 10.0)`), `spatial_missing`.

### Canonicalization & Trap Handling
- **Conservative Telex/VNI cleanup:** Normalizes unmerged Telex/VNI artifacts (`Haf Nooij`, `Quaanj 7`, `Loojc Thanhj`, `Ha2 No6i5`) validated against Vietnamese syllable structure & gazetteer vocabulary while protecting English/food/brand tokens and house/ward numbers (`P.6`, `Q3`).
- **Compound Hyphenated Names:** Protects `ba ria - vung tau`, `phan rang - thap cham`, and 2025 hyphenated wards (`xuan huong - da lat`, `van mieu - quoc tu giam`, etc.) prior to ` - ` segment splitting.
- **Street & Proper-Substring Protection:** Positional/prefix role resolution + whole-token matching avoids false contradictions on street/ward/province collisions (e.g., `"45 Bình Thuận, Quận 7, TP.HCM"`, `"123 Hai Bà Trưng, P.6, Q3"`).

## Calibration & Routing
- **Temperature Scaling (`T`):** `0.75` (fitted on `6,000` validation split).
- **Dual Thresholds:** `t_low = 0.38`, `t_high = 0.66`:
  - `risk_score < 0.38` -> `auto_approve` (`is_valid_fill = True`)
  - `0.38 <= risk_score < 0.66` -> `soft_confirm_zns` (returns Vietnamese `zns_confirm_message`)
  - `risk_score >= 0.66` -> `hard_block`

## Datasets & Evaluation (`eval_report_v3.json`)
- **Train / Val / Test (`seed = 301`):** `48,000 / 6,000 / 6,000` (`55%` valid matches, `45%` bad/risky fills), strictly zero overlap on normalized `(raw_text, goong_ward, goong_district, goong_province)` key.
- **Independent Holdout (`seed = 20261199`):** `600` pairs using separate phrasing templates, street pools, and chat/food strings, verified zero overlap against all `60,000` train/val/test pairs.

| Metric | 6k Test Split | 600 Independent Holdout |
| --- | --- | --- |
| ROC-AUC | `0.999866` | `1.000000` |
| PR-AUC | `0.999836` | `1.000000` |
| ECE (15 bins) | `0.005180` | `0.004418` |
| Accuracy @ `t_low` | `0.991833` | `1.000000` |
| Hard-Block Precision | `0.996992` | `1.000000` |
| Hallucination Recall | `0.999295` | `1.000000` |
| Contradiction Recall | `1.000000` | `1.000000` |
| Spatial Outlier Recall | `1.000000` | `1.000000` |
| Chat / Food Recall | `1.000000` | `1.000000` |
| Legacy <-> 2025 False Alarm Rate | `0.001818` | `0.000000` |

### CPU Latency (600 Holdout Calls, Clean Zero-PyTorch Runtime)
- **Mean:** `0.845 ms`
- **p50:** `0.8247 ms`
- **p95:** `1.3564 ms`
- **p99:** `1.6431 ms`

## Limitations
- Relies on the bundled Vietnam administrative gazetteer (`vn_admin_v3_converter.json`) for exact/2025 equivalence and ward centroids; newly created or renamed administrative units outside that snapshot require refreshing `vn_admin_v3_converter.json`.
- When `goong_lat`/`goong_lng` are omitted (`None`), `spatial_missing` is `1.0` and spatial outlier detection cannot verify physical coordinate plausibility.
