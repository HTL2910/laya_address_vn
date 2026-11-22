"""Reproducible stdlib release builder for Laya Address VN v3.0.0.

Builds into `release_dist/`:
  1. laya-address-vn-widget-v3.0.0-web.zip       (Way A v3 Drop-in Webcake Widget + Telex/Fuzzy + 2025 Converter + ONNX WASM)
  2. laya-address-vn-suite-v3.0.0-backend.zip    (Complete Backend Suite: Way B v2 + Core 2025 + Post-Goong Verifier + Slot Parser)
  3. laya_address_vn-3.0.0-py3-none-any.whl      (Unified PEP-427 Python Wheel exporting all v3 & v2 engines)
  4. laya-address-vn-ceiling-audit-v3.0.0.zip    (Offline Step 1 & Step 2 DEV Audit Toolkit)
  5. SHA256SUMS.txt                              (SHA-256 checksums for all v3.0.0 archives)
  6. RELEASE_NOTES_V3.md                         (Vietnamese Release Notes for GitHub Release v3.0.0)
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple
import zipfile

VERSION = "3.1.0"
ZIP_TIMESTAMP = (2026, 11, 22, 20, 50, 0)

ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = ROOT / "release_dist"

CORE_DIR = ROOT / "laya_v3_core"
WAY_B_DIR = ROOT / "way_b_backend_api_v2"
WAY_C_DIR = ROOT / "way_c_post_goong_v3"
SLOT_DIR = ROOT / "slot_parser_v3"
WIDGET_DIR = ROOT / "way_a_widget_v3"
SHARED_V2_DIR = ROOT / "shared_data"
SHARED_V3_DIR = ROOT / "shared_data_v3"
AUDIT_DIR = ROOT / "ceiling_audit"


def _add_bytes(
    zf: zipfile.ZipFile,
    arcname: str,
    data: bytes,
    records: List[Tuple[str, str, int]] | None = None,
) -> None:
    zinfo = zipfile.ZipInfo(filename=arcname, date_time=ZIP_TIMESTAMP)
    zinfo.compress_type = zipfile.ZIP_DEFLATED
    zinfo.external_attr = 0o644 << 16
    zf.writestr(zinfo, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    if records is not None:
        digest = hashlib.sha256(data).digest()
        b64 = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        records.append((arcname, f"sha256={b64}", len(data)))


def _add_file(
    zf: zipfile.ZipFile,
    src: Path,
    arcname: str,
    records: List[Tuple[str, str, int]] | None = None,
) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing release source file: {src}")
    _add_bytes(zf, arcname, src.read_bytes(), records=records)


def _verify_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "r") as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"Corrupted ZIP member in {path.name}: {bad}")


def build_widget_web_zip() -> Path:
    out_path = DIST_DIR / f"laya-address-vn-widget-v{VERSION}-web.zip"
    prefix = f"laya-address-vn-widget-v{VERSION}-web"

    with zipfile.ZipFile(out_path, "w") as zf:
        for folder in ("sdk", "demo", "dist"):
            base = WIDGET_DIR / folder
            for src in sorted(base.rglob("*")):
                if src.is_file() and "__pycache__" not in src.parts:
                    rel = src.relative_to(WIDGET_DIR).as_posix()
                    _add_file(zf, src, f"{prefix}/{rel}")
        for fname in ("package.json", "serve.py", "README.md", "LICENSE_vietnamadminunits.txt"):
            _add_file(zf, WIDGET_DIR / fname, f"{prefix}/{fname}")
        _add_file(zf, SHARED_V3_DIR / "PROVENANCE_V3.json", f"{prefix}/PROVENANCE_V3.json")

    _verify_zip(out_path)
    return out_path


def build_backend_suite_zip() -> Path:
    out_path = DIST_DIR / f"laya-address-vn-suite-v{VERSION}-backend.zip"
    prefix = f"laya-address-vn-suite-v{VERSION}-backend"

    quickstart = f"""# Laya Address VN v{VERSION} — Full Backend CPU Suite

Bộ backend hoàn chỉnh gồm 4 mô-đun độc lập (chạy thuần CPU + ONNX Runtime INT8/FP32, không cần GPU hay PyTorch):
1. `way_b_backend_api_v2/` — Cổng Tiền-Goong Hybrid + ONNX INT8 (`p99 = 1.097 ms`, port `8084`)
2. `laya_v3_core/` — Bộ sửa Telex/VNI + SymSpell 1-edit + Chuyển đổi 2025 Cũ <-> Mới + Haversine (`0.185 ms`)
3. `slot_parser_v3/` — Bóc tách 17 thẻ BIO (BiGRU + NumPy Viterbi CRF, ONNX INT8 `1.52 MB`, `p99 = 5.17 ms`)
4. `way_c_post_goong_v3/` — Cổng xác minh Hậu-Goong ONNX (`68.5 KB`, `p99 = 1.64 ms`, port `8085`) trả về 3 mức hành động (`auto_approve` / `soft_confirm_zns` / `hard_block`)

## Khởi chạy nhanh
```powershell
python -m venv .venv
.\\.venv\\Scripts\\pip.exe install -r way_c_post_goong_v3\\requirements.txt
# 1) Huấn luyện mô hình 2A cục bộ trên đơn thật (chỉ cần numpy/pandas/scikit-learn/onnxruntime):
python way_c_post_goong_v3\\train_local_2a.py --data dev_orders.csv --output-dir local_2a_artifacts
# 2) Khởi chạy Post-Goong Verifier API trỏ vào mô hình cục bộ (port 8085):
$env:LAYA_POST_GOONG_ARTIFACTS_DIR = "local_2a_artifacts"
cd way_c_post_goong_v3 && ..\\.venv\\Scripts\\uvicorn.exe server:app --port 8085
```
""".encode("utf-8")

    with zipfile.ZipFile(out_path, "w") as zf:
        # 1. laya_v3_core
        for rel in ("__init__.py", "telex_fuzzy.py", "converter_2025.py", "requirements.txt"):
            _add_file(zf, CORE_DIR / rel, f"{prefix}/laya_v3_core/{rel}")

        # 2. way_c_post_goong_v3 (v3.0 + v3.1 local 2A pipeline)
        for rel in (
            "__init__.py",
            "features.py",
            "features_2a.py",
            "onnx_export_pure.py",
            "train_local_2a.py",
            "verifier.py",
            "server.py",
            "requirements.txt",
            "MODEL_CARD.md",
            "artifacts/post_goong_verifier_v3.onnx",
            "artifacts/calibration.json",
            "artifacts/scaler.json",
            "artifacts/eval_report_v3.json",
            "artifacts/vn_admin_v3_converter.json",
        ):
            _add_file(zf, WAY_C_DIR / rel, f"{prefix}/way_c_post_goong_v3/{rel}")

        # 3. slot_parser_v3
        for rel in (
            "__init__.py",
            "tokenizer.py",
            "parser.py",
            "requirements.txt",
            "MODEL_CARD.md",
            "artifacts/slot_parser_v3_int8.onnx",
            "artifacts/crf_transitions.json",
            "artifacts/vocab.json",
            "artifacts/eval_report_v3.json",
        ):
            _add_file(zf, SLOT_DIR / rel, f"{prefix}/slot_parser_v3/{rel}")

        # 4. way_b_backend_api_v2
        for src in sorted((WAY_B_DIR / "app").rglob("*")):
            if src.is_file() and "__pycache__" not in src.parts and src.suffix != ".pyc":
                rel = src.relative_to(WAY_B_DIR).as_posix()
                _add_file(zf, src, f"{prefix}/way_b_backend_api_v2/{rel}")
        for fname in ("calibration.json", "laya_address_vn_int8.onnx", "vn_admin_gazetteer.json", "vocab.json"):
            _add_file(zf, WAY_B_DIR / "artifacts" / fname, f"{prefix}/way_b_backend_api_v2/artifacts/{fname}")
        _add_file(zf, WAY_B_DIR / "requirements.txt", f"{prefix}/way_b_backend_api_v2/requirements.txt")
        _add_file(zf, WAY_B_DIR / "MODEL_CARD.md", f"{prefix}/way_b_backend_api_v2/MODEL_CARD.md")

        # 5. shared_data_v3 converter & attribution
        for fname in ("vn_admin_v3_converter.json", "PROVENANCE_V3.json", "LICENSE_vietnamadminunits.txt"):
            _add_file(zf, SHARED_V3_DIR / fname, f"{prefix}/shared_data_v3/{fname}")
        _add_bytes(zf, f"{prefix}/QUICKSTART_SUITE_V3.md", quickstart)

    _verify_zip(out_path)
    return out_path


def _rewrite_way_b_for_wheel(filename: str, text: str) -> str:
    out = text.replace("from app.", "from laya_address_vn.")
    if filename == "engine.py":
        target = "BASE_DIR = Path(__file__).resolve().parent.parent"
        replacement = "BASE_DIR = Path(__file__).resolve().parent"
        if target not in out:
            raise RuntimeError("Unexpected engine.py BASE_DIR layout")
        out = out.replace(target, replacement, 1)
    elif filename == "heuristics.py":
        target = 'DEFAULT_GAZETTEER_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "vn_admin_gazetteer.json"'
        replacement = 'DEFAULT_GAZETTEER_PATH = Path(__file__).resolve().parent / "artifacts" / "vn_admin_gazetteer.json"'
        if target not in out:
            raise RuntimeError("Unexpected heuristics.py DEFAULT_GAZETTEER_PATH layout")
        out = out.replace(target, replacement, 1)
    elif filename == "server.py":
        out = out.replace('"app.server:app"', '"laya_address_vn.server:app"')
    return out


def _rewrite_core_converter_for_wheel(text: str) -> str:
    target = 'Path(__file__).resolve().parent.parent / "shared_data_v3" / "vn_admin_v3_converter.json"'
    replacement = 'Path(__file__).resolve().parent.parent / "post_goong" / "artifacts" / "vn_admin_v3_converter.json"'
    if target not in text:
        raise RuntimeError("Unexpected converter_2025.py DEFAULT_CONVERTER_PATH layout")
    return text.replace(target, replacement, 1)


def build_unified_wheel_v3() -> Path:
    dist_name = "laya_address_vn"
    out_path = DIST_DIR / f"{dist_name}-{VERSION}-py3-none-any.whl"
    dist_info = f"{dist_name}-{VERSION}.dist-info"

    init_py = f'''"""Laya Address VN v{VERSION} — Unified CPU Inference SDK.

Includes:
  - Pre-Goong Sufficiency Gate (Way B v2 ONNX INT8 + MIT Gazetteer)
  - Core Telex/VNI Repair, O(1) SymSpell Fuzzy & 2025 2-Tier Converter
  - BIO-NER Address Slot Parser (ONNX INT8 + NumPy Viterbi CRF)
  - Post-Goong Fill Verifier (ONNX + Spatial Haversine + Legacy/2025 Equivalence)
"""

from laya_address_vn.engine import (
    DEFAULT_ARTIFACTS_DIR,
    SUFFICIENCY_CLASSES,
    LayaAddressEngine,
    NoulDecision,
)
from laya_address_vn.heuristics import GazetteerIndex, evaluate_heuristic
from laya_address_vn.core import (
    SymSpellGazetteer,
    VietnamAdminV3Index,
    haversine_km,
    normalize_text,
    repair_telex_typos,
    strip_accents,
)
from laya_address_vn.post_goong import (
    FEATURE_NAMES_1B_24,
    FEATURE_NAMES_2A_51,
    FEATURE_NAMES_2A_V31,
    FeatureExtractor2A,
    PostGoongVerifier,
    export_lr_hgb_ensemble_to_onnx,
)
from laya_address_vn.slot_parser import AddressSlotParser, BIO_TAGS, ParsedAddress, SLOT_TYPES

__version__ = "{VERSION}"

__all__ = [
    "AddressSlotParser",
    "BIO_TAGS",
    "DEFAULT_ARTIFACTS_DIR",
    "FEATURE_NAMES_1B_24",
    "FEATURE_NAMES_2A_51",
    "FEATURE_NAMES_2A_V31",
    "FeatureExtractor2A",
    "GazetteerIndex",
    "LayaAddressEngine",
    "NoulDecision",
    "ParsedAddress",
    "PostGoongVerifier",
    "SLOT_TYPES",
    "SUFFICIENCY_CLASSES",
    "SymSpellGazetteer",
    "VietnamAdminV3Index",
    "__version__",
    "evaluate_heuristic",
    "export_lr_hgb_ensemble_to_onnx",
    "haversine_km",
    "normalize_text",
    "repair_telex_typos",
    "strip_accents",
]
'''

    metadata = f"""Metadata-Version: 2.1
Name: laya-address-vn
Version: {VERSION}
Summary: Complete Vietnamese address gate, 2025 2-tier converter, BIO-NER slot parser & post-Goong 2A verifier (CPU ONNX)
Home-page: https://github.com/HTL2910/laya_address_vn
License: MIT
Requires-Python: >=3.9
Requires-Dist: numpy>=1.26.0
Requires-Dist: onnxruntime>=1.19.0
Provides-Extra: server
Requires-Dist: fastapi>=0.110.0; extra == "server"
Requires-Dist: uvicorn>=0.27.0; extra == "server"
Requires-Dist: pydantic>=2.5.0; extra == "server"
Provides-Extra: train
Requires-Dist: scikit-learn>=1.5.0; extra == "train"
Requires-Dist: pandas>=2.2.0; extra == "train"
Requires-Dist: openpyxl>=3.1.0; extra == "train"
Description-Content-Type: text/markdown

# Laya Address VN v{VERSION} (`laya_address_vn`)

Unified pure-CPU Vietnamese address intelligence library (zero PyTorch, zero pickle, zero GPU required).

```python
from laya_address_vn import (
    LayaAddressEngine,
    AddressSlotParser,
    PostGoongVerifier,
    VietnamAdminV3Index,
)

gate = LayaAddressEngine()
slot = AddressSlotParser()
conv = VietnamAdminV3Index()
ver = PostGoongVerifier()

parsed = slot.parse("Anh Nam 0912345678 - 12/4/5 Lê Lợi, Phúc Xá, Ba Đình, Hà Nội (giao giờ HC)")
c2025 = conv.convert_legacy_to_2025("Phường Phúc Xá", "Quận Ba Đình", "Hà Nội")
v = ver.verify("Số 5 gần chợ, Tân Lập", "Xã Tân Lập", "Huyện Đồng Phú", "Tỉnh Bình Phước")
```
""".encode("utf-8")

    wheel_meta = """Wheel-Version: 1.0
Generator: laya-release-builder (3.0)
Root-Is-Purelib: true
Tag: py3-none-any
""".encode("utf-8")

    records: List[Tuple[str, str, int]] = []
    with zipfile.ZipFile(out_path, "w") as zf:
        # 1. Root laya_address_vn (Way B v2 engine + artifacts)
        _add_bytes(zf, f"{dist_name}/__init__.py", init_py.encode("utf-8"), records)
        for py_name in ("tokenizer.py", "heuristics.py", "engine.py", "server.py"):
            src_txt = (WAY_B_DIR / "app" / py_name).read_text(encoding="utf-8")
            rewritten = _rewrite_way_b_for_wheel(py_name, src_txt).encode("utf-8")
            _add_bytes(zf, f"{dist_name}/{py_name}", rewritten, records)
        _add_file(zf, WAY_B_DIR / "app" / "static" / "index.html", f"{dist_name}/static/index.html", records)
        for fname in ("calibration.json", "laya_address_vn_int8.onnx", "vn_admin_gazetteer.json", "vocab.json"):
            _add_file(zf, WAY_B_DIR / "artifacts" / fname, f"{dist_name}/artifacts/{fname}", records)

        # 2. laya_address_vn.core (Telex + SymSpell + 2025 Converter, reusing post_goong converter JSON)
        _add_file(zf, CORE_DIR / "__init__.py", f"{dist_name}/core/__init__.py", records)
        _add_file(zf, CORE_DIR / "telex_fuzzy.py", f"{dist_name}/core/telex_fuzzy.py", records)
        conv_txt = (CORE_DIR / "converter_2025.py").read_text(encoding="utf-8")
        _add_bytes(zf, f"{dist_name}/core/converter_2025.py", _rewrite_core_converter_for_wheel(conv_txt).encode("utf-8"), records)

        # 3. laya_address_vn.post_goong (Post-Goong Verifier ONNX + local 2A pipeline + shared vn_admin_v3_converter.json)
        for fname in ("__init__.py", "features.py", "features_2a.py", "onnx_export_pure.py", "train_local_2a.py", "verifier.py", "server.py"):
            _add_file(zf, WAY_C_DIR / fname, f"{dist_name}/post_goong/{fname}", records)
        for fname in ("post_goong_verifier_v3.onnx", "calibration.json", "scaler.json", "vn_admin_v3_converter.json"):
            _add_file(zf, WAY_C_DIR / "artifacts" / fname, f"{dist_name}/post_goong/artifacts/{fname}", records)

        # 4. laya_address_vn.slot_parser (BIO-NER INT8 ONNX + CRF transitions + NumPy Viterbi)
        for fname in ("__init__.py", "tokenizer.py", "parser.py"):
            _add_file(zf, SLOT_DIR / fname, f"{dist_name}/slot_parser/{fname}", records)
        for fname in ("slot_parser_v3_int8.onnx", "crf_transitions.json", "vocab.json"):
            _add_file(zf, SLOT_DIR / "artifacts" / fname, f"{dist_name}/slot_parser/artifacts/{fname}", records)

        # 5. PEP-427 dist-info
        _add_bytes(zf, f"{dist_info}/METADATA", metadata, records)
        _add_bytes(zf, f"{dist_info}/WHEEL", wheel_meta, records)
        _add_file(zf, SHARED_V3_DIR / "LICENSE_vietnamadminunits.txt", f"{dist_info}/LICENSE", records)

        record_lines = [f"{p},{h},{sz}" for p, h, sz in records]
        record_lines.append(f"{dist_info}/RECORD,,")
        _add_bytes(zf, f"{dist_info}/RECORD", ("\n".join(record_lines) + "\n").encode("utf-8"), None)

    _verify_zip(out_path)
    return out_path


def build_ceiling_audit_zip() -> Path:
    out_path = DIST_DIR / f"laya-address-vn-ceiling-audit-v{VERSION}.zip"
    prefix = f"laya-address-vn-ceiling-audit-v{VERSION}"

    audit_files = (
        "README.md",
        "requirements.txt",
        "schema_io.py",
        "post_goong_features.py",
        "v2_scorer.py",
        "ceiling_eval.py",
        "sample_300_for_labeling.py",
        "score_300_labels.py",
        "generate_demo_dev_data.py",
        "tests/test_ceiling_audit.py",
    )
    shared_files = (
        "vn_admin_gazetteer.json",
        "LICENSE_vietnamadminunits.txt",
        "PROVENANCE.json",
    )
    with zipfile.ZipFile(out_path, "w") as zf:
        for rel in audit_files:
            _add_file(zf, AUDIT_DIR / rel, f"{prefix}/ceiling_audit/{rel}")
        for rel in shared_files:
            _add_file(zf, SHARED_V2_DIR / rel, f"{prefix}/shared_data/{rel}")
    _verify_zip(out_path)
    return out_path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    built: List[Path] = [
        build_widget_web_zip(),
        build_backend_suite_zip(),
        build_unified_wheel_v3(),
        build_ceiling_audit_zip(),
    ]

    sums: Dict[str, str] = {}
    sum_lines: List[str] = []
    for p in built:
        digest = sha256_file(p)
        sums[p.name] = digest
        sum_lines.append(f"{digest}  {p.name}")

    (DIST_DIR / "SHA256SUMS.txt").write_text("\n".join(sum_lines) + "\n", encoding="utf-8")

    notes = f"""# Laya Address VN v{VERSION} — Real-Order 2A Trainer + Robust SlotParser Fallback

Bản nâng cấp `v{VERSION}` trực tiếp khắc phục hai điểm nghẽn phát hiện khi đánh giá trên **18.371 đơn thật** (giữ nguyên toàn bộ `v2.0` và các ONNX `v3.0`):

1. **`SlotParser v3.1` (`slot_parser_v3/parser.py`):**
   - Bổ sung **phone-guard tất định** (chặt 100% số điện thoại nằm cạnh số nhà không lọt vào `HNO` / `clean_address_for_goong`).
   - Bổ sung **subtractive + right-to-left 2025 gazetteer fallback** khi chuỗi không có dấu phẩy / không tiền tố / không dấu khiến CRF gắn toàn nhãn `O` (`used_fallback=True`), vẫn giữ chặn cứng cụm chat/món ăn & chuỗi quá ngắn.
2. **Local Real-Order `2A` Trainer & Pure-Stdlib ONNX Exporter (`way_c_post_goong_v3/`):**
   - `features_2a.py`: trích xuất `24` (`1B`) / `51` (`2A`) / `56` (`2A + 2025 equiv + Haversine`) đặc trưng phi khu vực, `0` mã tỉnh, `0` từ vựng PII.
   - `onnx_export_pure.py`: xuất thẳng ensemble `LogisticRegression + HistGradientBoostingClassifier` sang ONNX chuẩn (`ir_version=8`, `ai.onnx.ml` `TreeEnsembleRegressor`) **chỉ bằng Python stdlib** — hoàn toàn không cần `torch`, `onnx`, hay `skl2onnx` (`parity max_abs_diff < 1e-4`).
   - `train_local_2a.py`: 1 lệnh chạy offline trên dữ liệu DEV thật (`0` PII rời máy), đánh giá kiểm chéo thời gian + kiểm chứng đóng băng ngưỡng tương lai (`calib -> future`), xuất `post_goong_verifier_2a_local.onnx` + `calibration.json` (bật `allow_missing_ward` tự động nếu DB thiếu tên xã).
   - `verifier.py` + `server.py`: tự nhận diện schema `v3_43` lẫn `v31_2a_*` và biến môi trường `LAYA_POST_GOONG_ARTIFACTS_DIR`.

## Tệp phát hành đính kèm

| Tệp | Nội dung | Dung lượng |
|---|---|---:|
| `laya-address-vn-widget-v{VERSION}-web.zip` | **Bản Web Widget SDK (`@laya/address-vn-widget`)** — Custom Element `<laya-address-input>`, sửa Telex + SymSpell 1-edit, gợi ý Top-3 Xã/Phường 1-chạm, tự đổi 3 cấp cũ <-> 2 cấp 2025 + ONNX WASM (`0đ` server, `0` CDN) | `{(built[0].stat().st_size / (1024 * 1024)):.2f} MB` |
| `laya-address-vn-suite-v{VERSION}-backend.zip` | **Bản Backend Suite v{VERSION}** — Đầy đủ `way_b_backend_api_v2` + `laya_v3_core` + `slot_parser_v3` (có fallback + phone-guard) + `way_c_post_goong_v3` (kèm `train_local_2a.py` & pure ONNX exporter) | `{(built[1].stat().st_size / (1024 * 1024)):.2f} MB` |
| `laya_address_vn-{VERSION}-py3-none-any.whl` | **Unified Python Wheel** — `pip install` cho runtime (`numpy`, `onnxruntime`) hoặc `[server]`, `[train]` | `{(built[2].stat().st_size / (1024 * 1024)):.2f} MB` |
| `laya-address-vn-ceiling-audit-v{VERSION}.zip` | **Bộ đo trần offline (Bước 1–2)** — Kiểm chứng trên log đơn DEV thật, `0%` rò rỉ PII | `{(built[3].stat().st_size / 1024):.1f} KB` |
| `SHA256SUMS.txt` | Mã kiểm tra toàn vẹn SHA-256 | — |

## SHA-256 Checksums
```text
{chr(10).join(sum_lines)}
```
"""
    (DIST_DIR / "RELEASE_NOTES_V3.md").write_text(notes, encoding="utf-8")
    for p in built:
        print(f"BUILT {p.name:48s}  {p.stat().st_size:10,d} B  sha256={sums[p.name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
