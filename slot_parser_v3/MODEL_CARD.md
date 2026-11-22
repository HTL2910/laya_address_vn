# Model Card: `slot_parser_v3`

## Overview
`slot_parser_v3` is a compact Vietnamese e-commerce address and chat slot parser combining a Word + Char-CNN + 2-Layer Bidirectional GRU emission encoder (exported to INT8 ONNX) with an exact BIO-constrained Linear-CRF decoded at runtime via pure NumPy Viterbi.

Runtime inference (`AddressSlotParser`) requires zero PyTorch and zero pickle (`numpy` + `onnxruntime` CPUExecutionProvider only).

## Schema (`BIO_TAGS`, 17 Tags)
- `0`: `O`
- `1, 2`: `B-HNO`, `I-HNO` — house / alley / apartment unit numbers (`12/4/5`, `45/2B`, `số 10`, `P.702`)
- `3, 4`: `B-STR`, `I-STR` — street, alley, hamlet, residential unit, or building (`Nguyễn Trãi`, `đường Lê Lợi`, `ngõ 12`, `Thôn 3`, `Ấp 2`, `Chung cư HH1C`)
- `5, 6`: `B-WAR`, `I-WAR` — ward / commune / township / special zone (`Phường Bến Nghé`, `Xã Lộc Thạnh`, `x. Hồng Dương`, `Bến Nghé`)
- `7, 8`: `B-DIS`, `I-DIS` — district / county / town / city (`Quận 1`, `Q1`, `Q.1`, `Huyện Lộc Ninh`, `Lộc Ninh`, `TP. Thủ Đức`)
- `9, 10`: `B-PRO`, `I-PRO` — province / centrally-governed city (`TP.HCM`, `Hà Nội`, `Đồng Nai`, `Bình Phước`, `t. Bình Dương`)
- `11, 12`: `B-PER`, `I-PER` — recipient cue + name (`Anh Nam`, `Chị Trần Thị Hoa`, `Người nhận Lê Minh`)
- `13, 14`: `B-PHN`, `I-PHN` — phone number (`0912345678`, `0912.345.678`, `+84987654321`)
- `15, 16`: `B-NOTE`, `I-NOTE` — delivery notes (`giao giờ hành chính`, `gọi trước 15p`, `ship sau 5h`, `hàng dễ vỡ`)

## Architecture & Runtime Artifacts
- **Tokenizer**: Pure Python + NumPy whitespace/punctuation-aware `tokenize_words` + `SlotParserTokenizer` (`max_word_len=64`, `max_char_per_word=16`, `word_vocab_size=6583`, `char_vocab_size=206`).
- **Encoder**: Word embedding (`64`) + Char embedding (`24`) -> Char-CNN (`k=3`, `48` channels, masked max-pool) -> concat (`112`) -> 2-layer BiGRU (`hidden=96` per direction, `192` out) -> Linear (`17`).
- **Sequence Decoder**: Exact PyTorch Linear-CRF trained with forward-backward NLL loss (`AdamW`, `CosineAnnealingLR`, word dropout `0.06`, 8 epochs), exported as constrained `17x17` transitions + start/end vectors (`artifacts/crf_transitions.json`) and decoded at inference by pure NumPy `viterbi_decode`.
- **Artifacts** (`slot_parser_v3/artifacts/`):
  - `slot_parser_v3_int8.onnx` (`1,597,217` bytes / `1.5232 MB`, dynamic `QUInt8` quantized from `2,890,881` bytes FP32, `quantization_f1_delta = 0.0001`)
  - `crf_transitions.json`
  - `vocab.json`
  - `eval_report_v3.json`

## Training & Evaluation Data
- **Reference Admin Ontology**: `shared_data_v3/vn_admin_v3_records.json` (10,602 MIT-licensed Vietnam records covering legacy 3-tier and 2025 2-tier divisions).
- **Training Split (`seed=401`)**: `50,000` token-aligned synthetic Vietnamese e-commerce strings split `42,000` train / `4,000` val / `4,000` test, including full 3-tier, 2025 2-tier, short 2-part bare/prefixed addresses, rural hamlets/alleys, apartments, chat noise (`PER`, `PHN`, `NOTE`), unaccented/abbreviated/Telex-typo forms, and `15%` compositional non-address negatives.
- **Independent Holdout (`seed=20261201`)**: `500` unseen samples built from disjoint street, building, rural unit, person name, delivery note, lead-in, and negative chat pools.

## Audited Benchmark Results (`artifacts/eval_report_v3.json`)

| Metric | 4k Test Split | 500 Independent Holdout |
| --- | --- | --- |
| Token Accuracy | `0.9952` | `0.9858` |
| Macro Entity F1 | `0.9929` | `0.9770` |
| Micro Entity F1 | `0.9906` | `0.9747` |
| Clean-Address Match Rate | `0.9965` | `0.9560` |
| Negative False-Positive Rate | `0.0000` | `0.0000` |

### Independent 500-Case Holdout Per-Slot Entity F1
- `HNO`: `1.0000` (`support=234`)
- `STR`: `0.9677` (`support=311`)
- `WAR`: `0.9520` (`support=347`)
- `DIS`: `0.9638` (`support=348`)
- `PRO`: `0.9883` (`support=298`)
- `PER`: `0.9581` (`support=194`)
- `PHN`: `1.0000` (`support=205`)
- `NOTE`: `0.9858` (`support=177`)

### CPU Single-Item Latency (`AddressSlotParser.parse`, `intra_op_threads=1`)
- `mean`: `3.199 ms`
- `p50`: `2.971 ms`
- `p95`: `3.772 ms`
- `p99`: `5.171 ms`

## Intended Use & Limitations
- Designed as an upstream slot extractor producing clean `house_num + street, ward, district, province` queries (`clean_address_for_goong`) stripped of recipient names, phone numbers, and delivery instructions before geocoding or admin canonicalization.
- Truncates token sequences beyond `max_word_len = 64` (tail tokens beyond index 63 default to `"O"`).
- Ambiguous bare 2-part names that appear across both 3-tier districts and 2025 2-tier wards rely on training distribution priors.
