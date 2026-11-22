# Model Card — Way B v2 (`laya-address-vn-way-b-v2-int8`)

## 1. Overview
**Way B v2** is a lightweight, CPU-only Vietnamese address sufficiency and classification microservice built with **FastAPI**, **ONNX Runtime (INT8)**, and a deterministic **Tier-1 Vietnam Administrative Gazetteer** heuristic engine. It classifies any short or free-text Vietnamese string into three operational sufficiency classes while remaining strictly backward-compatible with v1 binary callers:

- `not_address` (`is_address = false`): garbage, numeric codes, chat/order questions, e-commerce products, food+place names, person names, narrative/weather/travel sentences, URLs, phones, emails, tracking IDs.
- `address_insufficient` (`is_address = false`): incomplete location inputs missing deliverable granularity (ward-only with prefix, province-only, district-only or district+province without ward/street, or bare granular fragments like `"ngõ 12"`, `"Thôn 7"`, `"đường Lê Lợi"`).
- `address_sufficient` (`is_address = true`): deliverable/actionable addresses including all 6 short valid Vietnamese writing styles (`"Lộc Thạnh, Lộc Ninh"`, `"Lộc Thạnh Lộc Ninh"`, `"loc thanh loc ninh"`, `"x.loc thanh h.loc ninh"`, `"Lộc Thạnh, Lộc Ninh, Bình Phước"`, `"Lộc Thạnh, Bình Phước"`, `"Phường Hồng Hà, Hà Nội"`), full urban/rural/apartment/deep-alley addresses, and delivery free-text containing customer names, phones, and shipper notes.

---

## 2. Dataset & Provenance
- **Administrative Source Data:** `../shared_data/vn_admin_full_records.json` & `../shared_data/vn_admin_gazetteer.json` derived from [`tranngocminhhieu/vietnamadminunits`](https://github.com/tranngocminhhieu/vietnamadminunits) (**MIT License**, Copyright (c) 2025 Tran Ngoc Minh Hieu).
  - **10,035** legacy 3-tier records (`danhmuchanhchinh.gso.gov.vn_ward_2025-07-18.csv`) across **63 provinces** and **656 districts** (**7,044** unique normalized ward names).
  - **3,321** post-2025 2-tier merged records (`2025_34-province-3221-ward_with_location.csv`).
- **Generated Corpus (`train/dataset_v2.py`):**
  - **75,000** unique/stratified samples (`seed = 202`), split **80% / 10% / 10%** (`60,000` train / `7,500` val / `7,500` test) with zero normalized-key (`norm_key`) overlap across splits:
    - `address_sufficient` (label `2`): **45.0%** (`27,000` train / `3,375` val / `3,375` test)
    - `address_insufficient` (label `1`): **25.0%** (`15,000` train / `1,875` val / `1,875` test)
    - `not_address` (label `0`): **30.0%** (`18,000` train / `2,250` val / `2,250` test)
- **Strict Anti-Leakage Guard:**
  - Every generated sample was filtered against BOTH `../benchmark/vn_address_benchmark.json` (90 cases) and `../benchmark_v2/independent_holdout_v2.json` (600 cases):
    - **0** exact normalized accent-stripped matches across all 75,000 samples.
    - **0** word 4-gram overlaps for all negative, chat, and delivery-freetext samples.
  - All 60,000 final training texts are recorded in `train/train_texts_manifest.txt`.

---

## 3. Architecture (`train/model_v2.py`)
`LayaDualGranularitySufficiencyModel` (**1,023,461** parameters, zero PyTorch/pickle dependency at runtime):
- **Tokenizer (`app/tokenizer.py`):** Pure Python + NumPy `DualGranularityTokenizer` (`word_vocab = 3,243` with 100% accented + unaccented coverage of all Vietnam admin syllables; `char_vocab = 118`; `max_word_len = 64`, `max_char_len = 160`).
- **Branch 1 (Word Context):** Factorized Word Embedding (`64 -> 160`) + 2-Layer Pre-Norm Transformer (`4` heads, `d_ff = 512`) + CLS / Attention / Masked Mean Pooling (`-> 192`).
- **Branch 2 (Character Morphology):** Multi-Scale Char-CNN (`kernels 3, 5, 7`, `64` channels each) + Masked Global Max & Mean Pooling (`-> 128`).
- **Dual Output Heads (`96`-dim shared trunk):**
  1. `logits` (`[B]`): binary sufficiency gate (`address_sufficient` vs rest)
  2. `sufficiency_logits` (`[B, 3]`): 3-class sufficiency (`not_address`, `address_insufficient`, `address_sufficient`)
- **Training:** Dynamic batch padding (`batch_size = 256`, `epochs = 8`, `CosineAnnealingLR`, joint `BCEWithLogitsLoss + CrossEntropyLoss`, `torch.set_num_threads(12)`, `581.54s` total wall time).

---

## 4. Calibration & Quantization (`artifacts/calibration.json`)
- **Dual Temperature Scaling (fitted on 7,500-sample `val` split):**
  - Binary gate temperature **`T = 1.25809`** (tuned threshold `0.520`)
  - 3-class sufficiency temperature **`T_suf = 1.28803`**
- **Calibration Metrics (PyTorch / FP32):**
  - **Val Binary:** Pre-cal ECE `0.00902` (NLL `0.06220`) -> Post-cal ECE **`0.00610`** (NLL `0.05938`, Acc `0.9789`)
  - **Val 3-Class:** Pre-cal ECE `0.01048` (NLL `0.07713`) -> Post-cal ECE **`0.00543`** (NLL `0.07227`, Acc `0.9760`)
  - **Test Binary:** Post-cal ECE **`0.00486`**, NLL `0.05889`, Brier `0.01628`, Acc **`0.9784`**, F1 `0.9760`
  - **Test 3-Class:** Post-cal ECE **`0.00452`**, NLL `0.07377`, Brier `0.03911`, Acc **`0.9753`**, Macro-F1 `0.9740`
- **ONNX FP32 vs Dynamic INT8 (`QUInt8`):**
  - **FP32 (`laya_address_vn_fp32.onnx`, opset 17):** `3.940 MB` (`4,131,453` bytes) — single self-contained file
  - **INT8 (`laya_address_vn_int8.onnx`):** **`1.789 MB`** (`1,876,241` bytes) — **54.59% size reduction**, single self-contained file
  - **INT8 Test Split Accuracy:** Binary Acc **`0.9787`** (`F1 = 0.9763`, `ECE = 0.00442`), 3-Class Acc **`0.9753`** (`Macro-F1 = 0.9739`, `ECE = 0.00498`, per-class F1: `not_address = 0.9905`, `address_insufficient = 0.9548`, `address_sufficient = 0.9765`)

---

## 5. CPU Performance & Benchmark Accuracy
Measured on `Intel(R) Core(TM) i5-14400` (`CPUExecutionProvider`, `ort_intra_op_threads = 2`):
- **FP32 Single-Sample CPU Latency:** `mean = 0.470 ms`, `p50 = 0.402 ms`, `p95 = 0.933 ms`, `p99 = 1.513 ms` (`2,125.6 req/s`)
- **INT8 Single-Sample CPU Latency:** `mean = 0.477 ms`, **`p50 = 0.425 ms`**, **`p95 = 0.893 ms`**, **`p99 = 1.122 ms`** (`2,096.5 req/s`)
- **INT8 Batch-32 CPU Throughput:** `7.912 ms / batch` (`0.2473 ms / req`, **`4,044.3 req/s`**)
- **Read-Only External Holdout Evaluation:**
  - `benchmark/vn_address_benchmark.json` (90 cases): `hybrid = 89/90 (98.89%)`, `model_only = 82/90 (91.11%)`
  - `benchmark_v2/independent_holdout_v2.json` (600 cases): `hybrid = 599/600 (99.83% 3-class, 99.83% binary)`, `model_only = 536/600 (89.33% 3-class, 91.00% binary)`

---

## 6. Known Limitations
- **Legacy District vs. 2025 Merged Ward Name Collisions:** Following the 2025 34-province merger, ~850 wards share exact bare names with legacy districts/provinces (e.g., `"Nà Hang, Tuyên Quang"`, `"Thanh Xuân, Hà Nội"`, `"Lộc Ninh, Bình Phước"`). Without an explicit ward prefix (`"Xã Nà Hang"` / `"Phường Thanh Xuân"`), Tier-1 conservatively classifies `<District/Merged-Name>, <Province>` as `address_insufficient`.
- **Single-Sentence Relocation Chatter (`NL08`):** Unformatted narrative sentences expressing past relocation across two cities (`"Mình mới chuyển từ Huế ra Hà Nội làm việc tháng trước"`) can occasionally trigger Tier-2 ONNX false positives when routed without explicit question/weather/product cues.
- **CPU Runtime Scope:** Designed for short customer/shipping strings (`<= 1,000` chars; model truncates at `64` word tokens / `160` characters).

---

## 7. Audited Cross-Evaluation, Leakage & Changelog (`benchmark_v2/`)

### N-Gram & Exact Leakage Audit (`../benchmark_v2/leakage_report.json`)
- **Way A v1 Baseline Note:** v1 contained 5 literal source leaks (`C01`, `NN02`, `NN03`, `NN09`, `NN10`); v2 eliminates all exact overlaps across both Way A v2 and Way B v2.
- **Way B v2 Manifest (`60,000` training texts in `train/train_texts_manifest.txt`):**
  - vs Old 90-Case Benchmark: **`0`** exact raw matches, **`0`** exact normalized matches (`word 4-gram` overlap: `44/90` = `48.89%`, `char 15-gram`: `56/90` = `62.22%`, strictly from shared GSO administrative unit phrases).
  - vs Independent 600-Case Holdout (`seed=20261122`): **`0`** exact raw matches, **`0`** exact normalized matches (`word 4-gram`: `220/600` = `36.67%`, `char 15-gram`: `256/600` = `42.67%`).

### Independent 600-Case Holdout — All 3 Modes (`eval_report_v2.json`)
Class order for 3×3 confusion matrix `[true][pred]`: `["not_address", "address_insufficient", "address_sufficient"]` (`200 / 120 / 280`).

| Mode | 3-Cls Acc | Macro-F1 | Binary Acc | Binary F1 | ShortBlk (6 styles) | InsufRec | GarbBlk | HardNegBlk | ECE (Bin / 3-Cls) | CPU Latency (p50 / p95 / p99) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `heuristic_only` | `99.67%` | `0.9958` | `99.83%` | `0.9982` | `0.56%` (`1/180`) | `100.00%` | `100.00%` | `100.00%` | `0.0144 / 0.0227` | `0.073 / 0.290 / 0.358 ms` |
| `model_only` | `89.33%` | `0.8898` | `91.00%` | `0.9097` | `4.44%` (`8/180`) | `93.33%` | `86.00%` | `76.00%` | `0.0660 / 0.0650` | `0.388 / 0.871 / 1.164 ms` |
| **`hybrid`** | **`99.83%`** | **`0.9980`** | **`99.83%`** | **`0.9982`** | **`0.56%` (`1/180`)** | **`100.00%`** | **`100.00%`** | **`100.00%`** | **`0.0080 / 0.0152`** | **`0.079 / 0.455 / 1.097 ms`** |

- **Hybrid 3×3 Confusion Matrix:** `[[200, 0, 0], [0, 120, 0], [0, 1, 279]]` (`FP=0`, `FN=1`).
- **Hybrid Per-Group Accuracy (3-Class / Binary):**
  - `full_address` (100): `100.0% / 100.0%`
  - `short_valid_address` (180): `99.44% / 99.44%` (per-style false-block: `no_prefix_comma 0.0%`, `no_prefix_no_comma 0.0%`, `unaccented_lower 0.0%`, `abbreviated 0.0%`, `ward_district_province 0.0%`, `ward_province_2tier 3.3%` — single ambiguous `"Nà Hang, Tuyên Quang"`)
  - `insufficient_address` (120): `100.0% / 100.0%`
  - `garbage_and_chat` (100): `100.0% / 100.0%` (`11/11` mandatory garbage strings blocked)
  - `hard_negative` (100): `100.0% / 100.0%`
- **Old 90-Case & De-leaked 85-Case Benchmark (`hybrid`):**
  - Full 90: **`89/90 (98.89%)`** | De-leaked 85 (excluding `C01, NN02, NN03, NN09, NN10`): **`84/85 (98.82%)`** (up from v1 `86/90`).

### MIT Gazetteer Provenance
- Source: `tranngocminhhieu/vietnamadminunits` (MIT License, Copyright (c) 2025 Tran Ngoc Minh Hieu), GSO `2025-07-18` (`sha256: bad8474f...`) + 2025 34-province (`sha256: cf103923...`).
- Bundled `artifacts/vn_admin_gazetteer.json`: `SHA-256 = 11096369768889e62dc9b11f7065d9a414452207511ad4c87ad29c406732cc79` (`221,796` bytes).
- Also note: actual production delivery-loss/return ratio (`<= 1.5`) and `>= 30` extra caught returns must be validated on the user's private 34,263-order dataset since v2 training/holdout data is 100% synthetic.

### 8. Changelog vs. v1 (`way_b_backend_api`)
1. Added dual-output 3-class `sufficiency_logits` (`not_address`, `address_insufficient`, `address_sufficient`) + `sufficiency` / `sufficiency_probs` / `sufficiency_score` fields in `LayaAddressEngine.decide()` and FastAPI responses while keeping all v1 fields & ONNX input tensor names (`word_ids, attention_mask, char_ids`) backward-compatible.
2. Upgraded `app/heuristics.py` with MIT GSO+2025 hierarchical gazetteer matching (`63` provinces / `656` districts / `7,044` wards), cutting short valid address false blocks from `40.1%` (v1) to `0.56%` (v2) while maintaining `100%` insufficient & garbage blocking.
3. Expanded word vocab (`1,541 -> 3,243`) for 100% accented + unaccented Vietnamese administrative syllable coverage and tightened hard-negative training (`bánh mì Sài Gòn`, `mực Cà Mau`, `Cho mình 3 chai nướng BBQ`, person/street collisions).
4. Enforced strict anti-leakage filtering (`0` exact matches against eval sets), removed `.pt` checkpoints from `app/`, removed external CDN links from `app/static/index.html`, and pinned all 40 packages in `requirements.txt`.
