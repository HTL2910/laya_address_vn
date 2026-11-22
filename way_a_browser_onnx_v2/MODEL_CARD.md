# Model Card — Laya Address VN v2 (Way A: Client-Side Browser ONNX)

## Overview
`Laya Address VN v2` is a zero-server (`0đ` backend inference cost) Vietnamese address validation and sufficiency classification system running 100% inside the browser via vendored `onnxruntime-web` (WASM) + a compact GSO/2025 administrative gazetteer.

It combines two complementary tiers:
1. **Tier 1 (`web/heuristics.js`):** Deterministic hierarchical lookup against `vn_admin_gazetteer.json` (63 provinces, 656 districts, 7,044 unique wards, legacy 3-tier + 2025 2-tier edges) with span-disjointness verification, Vietnamese person-name guards, and food/order/chat anti-cue guards.
2. **Tier 2 (`web/model/laya_address_vn.onnx`):** `LayaAddressSufficiencyCharCNN`, a dual-output character-level convolutional network producing both a binary delivery-sufficiency gate (`logit`) and a 3-class sufficiency prediction (`sufficiency_logits`: `not_address`, `address_insufficient`, `address_sufficient`).

## Training Data & Licensing Provenance
- **Administrative Source Data:** `../shared_data/vn_admin_full_records.json` and `../shared_data/vn_admin_gazetteer.json`, derived from `tranngocminhhieu/vietnamadminunits` under the **MIT License** (`Copyright (c) 2025 Tran Ngoc Minh Hieu`; see `../shared_data/LICENSE_vietnamadminunits.txt` and `../shared_data/PROVENANCE.json`).
  - Legacy 3-tier records: `10,035` rows (`danhmuchanhchinh.gso.gov.vn_ward_2025-07-18.csv`)
  - Post-2025 2-tier merged records: `3,321` rows (`2025_34-province-3221-ward_with_location.csv`)
- **Synthetic Dataset (`train/dataset_v2.py`):**
  - `90,000` samples (`seed = 101`), stratified 80% / 10% / 10% (`train = 71,999`, `val = 9,000`, `test = 9,001`).
  - Class balance: `45.0%` `address_sufficient` (40,500), `25.0%` `address_insufficient` (22,500), `30.0%` `not_address` (27,000).
  - Covers all 6 short valid Vietnamese writing styles (comma-separated, no-comma, unaccented lowercase, abbreviated `x./p./tt./q./h.`, 3-tier, and 2-tier), full urban/rural/apartment/deep-alley/delivery-freetext addresses, insufficient fragments (`Xã ...`, `Hà Nội`, `Thanh Xuân, Hà Nội`, `ngõ 12`, `Thôn 7`, etc.), and hard negatives (short garbage, customer orders/chat, regional foods containing VN place names, Vietnamese person-name collisions, structured IDs/phones/dates/prices, vague non-addresses).
- **Strict Anti-Leakage:**
  - All generated samples were filtered against both `../benchmark/vn_address_benchmark.json` and `../benchmark_v2/independent_holdout_v2.json` to reject any normalized accent-stripped exact match, and to reject any negative/chat/freetext sample sharing a word 4-gram with any eval string.
  - All `71,999` final training texts are logged in `train/train_texts_manifest.txt`.

## Model Architecture & Export
- **Class:** `LayaAddressSufficiencyCharCNN` (`278,436` parameters)
- **Input:** `input_ids` (`int64`, shape `[B, 128]`, vocabulary size `123`)
- **Backbone:** `Embedding(123, 64)` -> parallel `Conv1d(k=2, 3, 5, 7)` + 2-layer contextual `Conv1d(k=3 -> 5)` -> concatenated `GlobalMaxPool` + `GlobalMeanPool` (`480` dims) -> `Linear(480, 320) -> ReLU -> Dropout(0.22) -> Linear(320, 128) -> ReLU -> Dropout(0.22)`
- **Outputs:**
  - `logit` (`[B, 1]`): binary gate (`address_sufficient` vs rest)
  - `sufficiency_logits` (`[B, 3]`): 3-class `["not_address", "address_insufficient", "address_sufficient"]`
- **ONNX Opset:** `14`, self-contained single protobuf (no `.onnx.data` sidecar, no `.pt` runtime artifacts)
- **ONNX Operators:** `Concat`, `Conv`, `Gather`, `Gemm`, `ReduceMax`, `ReduceMean`, `Relu`, `Transpose`
- **Torch vs ONNX Parity:** `max_abs_diff = 3.81e-05` (`< 1e-4`)

## Artifact Sizes (`web/model/`)
| File | Size |
| --- | --- |
| `laya_address_vn.onnx` | `1,118,184` bytes (`1,092.0 KB`) |
| `vn_admin_gazetteer.json` | `221,796` bytes (`216.6 KB`) |
| `calibration.json` | `3,084` bytes (`3.0 KB`) |
| `vocab.json` | `1,959` bytes (`1.9 KB`) |
| **Total `web/model/*`** | **`1,345,023` bytes (`1,313.5 KB` / `1.283 MB`)** |

## Calibration & Evaluation Metrics (Validation / Test Splits)
- **Binary Gate Calibration:** `temperature T = 1.61869`, `threshold = 0.5000` (tuned on validation: short-address false-block rate = `3.90%`, garbage block rate = `99.78%`)
- **3-Class Sufficiency Calibration:** `sufficiency_temperature T_suf = 1.62280`

| Head & Split | Pre-Cal ECE | Post-Cal ECE | Post-Cal NLL | Post-Cal Brier | Accuracy | F1 / Macro-F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Binary Gate (Val, n=9,000) | `0.00767` | `0.00659` | `0.04459` | `0.01289` | `0.98289` | `0.98097` |
| Binary Gate (Test, n=9,001) | `0.00659` | `0.00698` | `0.03966` | `0.01190` | `0.98356` | `0.98173` |
| 3-Class Sufficiency (Val, n=9,000) | `0.00692` | `0.00285` | `0.05272` | `0.02926` | `0.98067` | `0.97973` |
| 3-Class Sufficiency (Test, n=9,001) | `0.00586` | `0.00365` | `0.04702` | `0.02668` | `0.98222` | `0.98134` |

## CPU ONNX Runtime Single-Item Latency
Measured with single-threaded `onnxruntime==1.32.0` (`CPUExecutionProvider`, `intra_op_num_threads=1`, 300 runs after 25 warmups):
- **p50:** `0.255 ms`
- **p95:** `0.514 ms`
- **p99:** `0.638 ms`

## Vendored Browser Runtime
- `onnxruntime-web@1.20.1` vendored locally under `web/vendor/onnxruntime-web/` with `VERSION.txt` and `SHA256SUMS.txt` (zero external CDN dependencies).

## Honest Limitations
1. **Short Regional Food Names in Pure ONNX Mode:** Because the Char-CNN processes character n-grams without a full Vietnamese lexical dictionary, uncontextualized regional food phrases ending in real city names (e.g., `"bánh mì Sài Gòn"` alone) can receive high address scores from the neural network alone; in the default **Hybrid 2-Tier** mode, Tier-1 food/product/chat anti-cues accurately reject these as `not_address`.
2. **Unprefixed Single-Word Ambiguity:** Single-token inputs matching a commune or non-canonical short city alias (e.g., `"Vinh"`, `"Xom"`, `"Jean"`) are intentionally rejected or kept cautious unless accompanied by an admin prefix (`TP. Vinh`) or a verified parent/child span.
3. **Truncation at 128 Characters:** The ONNX Char-CNN inspects the first `128` normalized characters (`max_length = 128`). Extremely long freetext delivery notes exceeding 128 characters before the administrative suffix depend on Tier-1 gazetteer extraction over the full string.
4. **Administrative Boundary Currency:** Administrative units reflect GSO `2025-07-18` and the `2025` 34-province merger snapshot; future municipal boundary changes require regenerating `vn_admin_gazetteer.json` and retraining.

## Audited Cross-Evaluation & Leakage Report (`benchmark_v2/`)

### 1. N-Gram & Exact Leakage Audit (`../benchmark_v2/leakage_report.json`)
- **v1 (`way_a_browser_onnx/train/dataset.py`) Baseline Audit:** Found **5 literal source leaks** matching the old 90-case benchmark (`C01`, `NN02`, `NN03`, `NN09`, `NN10`). All 5 were eliminated in v2.
- **v2 Manifest Audit (`71,999` train samples in `train/train_texts_manifest.txt`):**
  - vs Old 90-Case Benchmark: **`0`** exact raw matches, **`0`** exact normalized matches (`word 4-gram` overlap: `46/90` = `51.11%`, `char 15-gram`: `61/90` = `67.78%`, strictly from shared administrative unit names in the public GSO catalog).
  - vs Independent 600-Case Holdout (`seed=20261122`): **`0`** exact raw matches, **`0`** exact normalized matches (`word 4-gram`: `257/600` = `42.83%`, `char 15-gram`: `275/600` = `45.83%`).

### 2. Independent 600-Case Holdout — All 3 Modes (`eval_report_v2.json`)
Class order for 3×3 confusion matrix `[true][pred]`: `["not_address", "address_insufficient", "address_sufficient"]` (`200 / 120 / 280`).

| Mode | 3-Cls Acc | Macro-F1 | Binary Acc | Binary F1 | ShortBlk (6 styles) | InsufRec | GarbBlk | HardNegBlk | ECE (Bin / 3-Cls) | CPU Latency (p50 / p95 / p99) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `heuristic_only` | `96.00%` | `0.9485` | `99.83%` | `0.9982` | `0.56%` (`1/180`) | `98.33%` | `100.00%` | `100.00%` | `0.0142 / 0.0338` | `0.016 / 0.071 / 0.249 ms` |
| `model_only` | `91.17%` | `0.9048` | `93.50%` | `0.9342` | `1.67%` (`3/180`) | `93.33%` | `91.00%` | `81.00%` | `0.0522 / 0.0488` | `0.282 / 0.320 / 0.561 ms` |
| **`hybrid`** | **`98.33%`** | **`0.9791`** | **`99.67%`** | **`0.9964`** | **`0.56%` (`1/180`)** | **`100.00%`** | **`100.00%`** | **`99.00%`** | **`0.0147 / 0.0157`** | **`0.015 / 0.295 / 0.381 ms`** |

- **Hybrid 3×3 Confusion Matrix:** `[[191, 8, 1], [0, 120, 0], [0, 1, 279]]` (`FP=1`, `FN=1`).
- **Hybrid Per-Group Accuracy (3-Class / Binary):**
  - `full_address` (100): `100.0% / 100.0%`
  - `short_valid_address` (180): `99.44% / 99.44%` (per-style false-block: `no_prefix_comma 0.0%`, `no_prefix_no_comma 0.0%`, `unaccented_lower 0.0%`, `abbreviated 0.0%`, `ward_district_province 0.0%`, `ward_province_2tier 3.3%` — single ambiguous `"Nà Hang, Tuyên Quang"`)
  - `insufficient_address` (120): `100.0% / 100.0%`
  - `garbage_and_chat` (100): `95.0% / 100.0%` (`11/11` mandatory garbage strings blocked)
  - `hard_negative` (100): `96.0% / 99.0%`
- **Old 90-Case & De-leaked 85-Case Benchmark (`hybrid`):**
  - Full 90: **`89/90 (98.89%)`** | De-leaked 85 (excluding `C01, NN02, NN03, NN09, NN10`): **`84/85 (98.82%)`** (up from v1 `87/90`).

### 3. Provenance & SHA-256 Verification
- **Gazetteer (`vn_admin_gazetteer.json`):** `SHA-256 = 11096369768889e62dc9b11f7065d9a414452207511ad4c87ad29c406732cc79` (`221,796` bytes; MIT `tranngocminhhieu/vietnamadminunits`).
- **Vendored `onnxruntime-web@1.20.1` (`web/vendor/onnxruntime-web/SHA256SUMS.txt`):**
  - `ort.min.js`: `be6e560b64c03c99252eedc0e1989e9e51e44d9f191e7655c9bf011bf9f576c8`
  - `ort.wasm.min.js`: `e66568724f8848e57cc2a56e4bea3a5b86ce3ff81b2da11eac8ed7ab02b27bd4`
  - `ort-wasm-simd-threaded.wasm`: `207d02be4591c156b0a98f024f3d58005b5b04c92274d759fb390338c63559ea`

### 4. Additional Operational Limitations
- **Real Customer Order Distribution:** Both training (`seed=101`) and holdout (`seed=20261122`) use synthetically generated templates over GSO+2025 names; actual false-block / return-catch ratio (`<= 1.5` lost-delivery-per-caught-return) and `>= 30` extra caught returns must be validated on the user's private 34,263-order dataset.
- **Unlisted Rural Hamlets & Heavy Typos:** Unprefixed short addresses containing typos in ward/district names or unofficial hamlet names fall back to Tier-2 Char-CNN (`91.17%` 3-class / `93.50%` binary standalone), which is less resilient than Tier-1 exact gazetteer lookup.

### 5. Changelog vs. v1 (`way_a_browser_onnx`)
1. Added 3-class `sufficiency_logits` output head alongside backward-compatible `logit` gate (`not_address`, `address_insufficient`, `address_sufficient`).
2. Integrated MIT GSO+2025 administrative gazetteer (`63` provinces / `656` districts / `7,044` wards) into `web/heuristics.js` with span-disjoint matching, fixing short unprefixed/no-comma/unaccented false blocks (`52.5%` pass in v1 → `99.44%` pass in v2).
3. Fixed `"thanh pho"` (`pho` street collision) and product-unit street collisions (`"66 Phan Chu Trinh"`, `"79/26 Ly Tu Trong"`).
4. Eliminated all 5 literal benchmark leaks present in v1 (`C01, NN02, NN03, NN09, NN10`) and removed `.pt` checkpoint persistence from runtime paths.
5. Vendored `onnxruntime-web@1.20.1` locally with SHA-256 verification (zero external CDN requests) and pinned all Python packages in `requirements.txt`.
