"""Reproducible stdlib release builder for Laya Address VN v2.0.0.

Builds into `release_dist/`:
  1. laya-address-vn-way-a-v2.0.0-web.zip        (Browser ONNX WASM standalone bundle, 0đ server)
  2. laya-address-vn-way-b-v2.0.0-backend.zip    (FastAPI CPU INT8 ONNX standalone bundle)
  3. laya_address_vn-2.0.0-py3-none-any.whl      (PEP-427 Python wheel exporting LayaAddressEngine)
  4. laya-address-vn-ceiling-audit-v2.0.0.zip    (Offline Step 1 & Step 2 DEV audit kit)
  5. SHA256SUMS.txt                               (SHA-256 checksums for all 4 archives)
  6. RELEASE_NOTES.md                             (Vietnamese release notes for GitHub Releases)
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple
import zipfile

VERSION = "2.0.0"
ZIP_TIMESTAMP = (2026, 11, 22, 17, 25, 0)

ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = ROOT / "release_dist"
WAY_A_DIR = ROOT / "way_a_browser_onnx_v2"
WAY_B_DIR = ROOT / "way_b_backend_api_v2"
SHARED_DIR = ROOT / "shared_data"
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
        raise FileNotFoundError(f"Missing release asset source file: {src}")
    _add_bytes(zf, arcname, src.read_bytes(), records=records)


def _verify_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "r") as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"Corrupted ZIP member in {path.name}: {bad}")


def build_way_a_web_zip() -> Path:
    out_path = DIST_DIR / f"laya-address-vn-way-a-v{VERSION}-web.zip"
    prefix = f"laya-address-vn-way-a-v{VERSION}-web"

    quickstart = f"""# Laya Address VN v{VERSION} — Way A (Bản dùng trực tiếp trên Web)

Chạy **100% trong trình duyệt của khách** bằng ONNX Runtime WASM (`onnxruntime-web@1.20.1` đã vendor sẵn, 0 CDN, `0đ` chi phí server).

## 1. Test nhanh trên máy
```bash
python serve.py
# Mở trình duyệt: http://127.0.0.1:8083
```

## 2. Nhúng vào Web / Form Webcake
1. Upload toàn bộ thư mục `web/` (giữ nguyên `model/`, `vendor/onnxruntime-web/`, `heuristics.js`).
2. Nhúng script cục bộ trong trang đặt hàng (trước bước gọi Goong Auto-Fill):
```html
<script src="/path/to/web/vendor/onnxruntime-web/ort.wasm.min.js"></script>
<script src="/path/to/web/heuristics.js"></script>
<script>
  async function initLayaGate() {{
    const [vocab, calib, gaz] = await Promise.all([
      fetch('/path/to/web/model/vocab.json').then(r => r.json()),
      fetch('/path/to/web/model/calibration.json').then(r => r.json()),
      fetch('/path/to/web/model/vn_admin_gazetteer.json').then(r => r.json()),
    ]);
    window.LayaHeuristics.initGazetteer(gaz);
    window.ort.env.wasm.wasmPaths = '/path/to/web/vendor/onnxruntime-web/';
    const session = await window.ort.InferenceSession.create('/path/to/web/model/laya_address_vn.onnx', {{
      executionProviders: ['wasm'],
    }});
    return {{ vocab, calib, session }};
  }}
</script>
```
3. Tham khảo logic hybrid đầy đủ Tier-1 (`LayaHeuristics.evaluateTier1`) -> Tier-2 (`session.run`) ngay trong `web/app.js`.
""".encode("utf-8")

    with zipfile.ZipFile(out_path, "w") as zf:
        web_root = WAY_A_DIR / "web"
        for src in sorted(web_root.rglob("*")):
            if src.is_file():
                rel = src.relative_to(WAY_A_DIR).as_posix()
                _add_file(zf, src, f"{prefix}/{rel}")
        _add_file(zf, WAY_A_DIR / "serve.py", f"{prefix}/serve.py")
        _add_file(zf, WAY_A_DIR / "MODEL_CARD.md", f"{prefix}/MODEL_CARD.md")
        _add_file(zf, SHARED_DIR / "LICENSE_vietnamadminunits.txt", f"{prefix}/LICENSE_vietnamadminunits.txt")
        _add_file(zf, SHARED_DIR / "PROVENANCE.json", f"{prefix}/PROVENANCE.json")
        _add_bytes(zf, f"{prefix}/QUICKSTART_WEB.md", quickstart)

    _verify_zip(out_path)
    return out_path


def build_way_b_backend_zip() -> Path:
    out_path = DIST_DIR / f"laya-address-vn-way-b-v{VERSION}-backend.zip"
    prefix = f"laya-address-vn-way-b-v{VERSION}-backend"

    quickstart = f"""# Laya Address VN v{VERSION} — Way B (Bản Backend CPU FastAPI + ONNX INT8)

Dịch vụ CPU thuần (`onnxruntime` INT8 `1.789 MB`, `p99 ~ 1.1 ms`, không cần PyTorch hay GPU).

## 1. Cài đặt & chạy server FastAPI
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate  |  Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.server:app --host 0.0.0.0 --port 8084
```

## 2. Gọi API `POST /v1/decide`
```bash
curl -X POST http://127.0.0.1:8084/v1/decide ^
  -H "Content-Type: application/json" ^
  -d "{{\\"text\\": \\"Lộc Thạnh, Lộc Ninh\\", \\"mode\\": \\"hybrid\\"}}"
```

## 3. Gọi trực tiếp trong Python
```python
from app.engine import LayaAddressEngine

engine = LayaAddressEngine("artifacts")
res = engine.decide("Lộc Thạnh, Lộc Ninh", mode="hybrid")
print(res.is_address, res.sufficiency, res.probability)
```
""".encode("utf-8")

    backend_artifacts = (
        "calibration.json",
        "laya_address_vn_int8.onnx",
        "vn_admin_gazetteer.json",
        "vocab.json",
    )

    with zipfile.ZipFile(out_path, "w") as zf:
        app_root = WAY_B_DIR / "app"
        for src in sorted(app_root.rglob("*")):
            if src.is_file() and "__pycache__" not in src.parts and src.suffix != ".pyc":
                rel = src.relative_to(WAY_B_DIR).as_posix()
                _add_file(zf, src, f"{prefix}/{rel}")
        for fname in backend_artifacts:
            _add_file(zf, WAY_B_DIR / "artifacts" / fname, f"{prefix}/artifacts/{fname}")
        _add_file(zf, WAY_B_DIR / "requirements.txt", f"{prefix}/requirements.txt")
        _add_file(zf, WAY_B_DIR / "MODEL_CARD.md", f"{prefix}/MODEL_CARD.md")
        _add_file(zf, SHARED_DIR / "LICENSE_vietnamadminunits.txt", f"{prefix}/LICENSE_vietnamadminunits.txt")
        _add_file(zf, SHARED_DIR / "PROVENANCE.json", f"{prefix}/PROVENANCE.json")
        _add_bytes(zf, f"{prefix}/QUICKSTART_BACKEND.md", quickstart)

    _verify_zip(out_path)
    return out_path


def _rewrite_for_wheel(filename: str, text: str) -> str:
    """Adapt `app.*` module imports & parent artifact paths for installed `laya_address_vn` wheel."""
    out = text.replace("from app.", "from laya_address_vn.")
    if filename == "engine.py":
        target = "BASE_DIR = Path(__file__).resolve().parent.parent"
        replacement = "BASE_DIR = Path(__file__).resolve().parent"
        if target not in out:
            raise RuntimeError("Unexpected engine.py BASE_DIR layout during wheel rewrite")
        out = out.replace(target, replacement, 1)
    elif filename == "heuristics.py":
        target = 'DEFAULT_GAZETTEER_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "vn_admin_gazetteer.json"'
        replacement = 'DEFAULT_GAZETTEER_PATH = Path(__file__).resolve().parent / "artifacts" / "vn_admin_gazetteer.json"'
        if target not in out:
            raise RuntimeError("Unexpected heuristics.py DEFAULT_GAZETTEER_PATH layout during wheel rewrite")
        out = out.replace(target, replacement, 1)
    elif filename == "server.py":
        out = out.replace('"app.server:app"', '"laya_address_vn.server:app"')
    return out


def build_way_b_wheel() -> Path:
    dist_name = "laya_address_vn"
    out_path = DIST_DIR / f"{dist_name}-{VERSION}-py3-none-any.whl"
    dist_info = f"{dist_name}-{VERSION}.dist-info"

    init_py = f'''"""Laya Address VN v{VERSION} — CPU ONNX INT8 & MIT Gazetteer Sufficiency Engine."""

from laya_address_vn.engine import (
    DEFAULT_ARTIFACTS_DIR,
    SUFFICIENCY_CLASSES,
    LayaAddressEngine,
    NoulDecision,
)
from laya_address_vn.heuristics import GazetteerIndex, evaluate_heuristic

__version__ = "{VERSION}"

__all__ = [
    "DEFAULT_ARTIFACTS_DIR",
    "GazetteerIndex",
    "LayaAddressEngine",
    "NoulDecision",
    "SUFFICIENCY_CLASSES",
    "__version__",
    "evaluate_heuristic",
]
'''

    metadata = f"""Metadata-Version: 2.1
Name: laya-address-vn
Version: {VERSION}
Summary: Vietnamese address gate & 3-class sufficiency classifier (ONNX INT8 + MIT VN Gazetteer)
Home-page: https://github.com/HTL2910/laya_address_vn
License: MIT (Gazetteer)
Requires-Python: >=3.9
Requires-Dist: numpy>=1.26.0
Requires-Dist: onnxruntime>=1.19.0
Provides-Extra: server
Requires-Dist: fastapi>=0.110.0; extra == "server"
Requires-Dist: uvicorn>=0.27.0; extra == "server"
Requires-Dist: pydantic>=2.5.0; extra == "server"
Description-Content-Type: text/markdown

# Laya Address VN (`laya_address_vn` v{VERSION})

CPU-only Vietnamese address sufficiency gate backed by ONNX Runtime INT8 (`1.789 MB`) and the MIT-licensed Vietnam Administrative Gazetteer (`63` provinces, `656` districts, `7,044` wards).

```python
from laya_address_vn import LayaAddressEngine

engine = LayaAddressEngine()
res = engine.decide("Lộc Thạnh, Lộc Ninh", mode="hybrid")
print(res.is_address, res.sufficiency, res.probability)
```

Optional FastAPI server:
```bash
pip install "laya_address_vn-{VERSION}-py3-none-any.whl[server]"
uvicorn laya_address_vn.server:app --host 0.0.0.0 --port 8084
```
""".encode("utf-8")

    wheel_meta = """Wheel-Version: 1.0
Generator: laya-release-builder (1.0)
Root-Is-Purelib: true
Tag: py3-none-any
""".encode("utf-8")

    records: List[Tuple[str, str, int]] = []
    with zipfile.ZipFile(out_path, "w") as zf:
        _add_bytes(zf, f"{dist_name}/__init__.py", init_py.encode("utf-8"), records)
        for py_name in ("tokenizer.py", "heuristics.py", "engine.py", "server.py"):
            src_txt = (WAY_B_DIR / "app" / py_name).read_text(encoding="utf-8")
            rewritten = _rewrite_for_wheel(py_name, src_txt).encode("utf-8")
            _add_bytes(zf, f"{dist_name}/{py_name}", rewritten, records)

        static_html = WAY_B_DIR / "app" / "static" / "index.html"
        _add_file(zf, static_html, f"{dist_name}/static/index.html", records)

        for fname in ("calibration.json", "laya_address_vn_int8.onnx", "vn_admin_gazetteer.json", "vocab.json"):
            _add_file(zf, WAY_B_DIR / "artifacts" / fname, f"{dist_name}/artifacts/{fname}", records)

        _add_bytes(zf, f"{dist_info}/METADATA", metadata, records)
        _add_bytes(zf, f"{dist_info}/WHEEL", wheel_meta, records)
        _add_file(zf, SHARED_DIR / "LICENSE_vietnamadminunits.txt", f"{dist_info}/LICENSE", records)

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
            _add_file(zf, SHARED_DIR / rel, f"{prefix}/shared_data/{rel}")

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
        build_way_a_web_zip(),
        build_way_b_backend_zip(),
        build_way_b_wheel(),
        build_ceiling_audit_zip(),
    ]

    sums: Dict[str, str] = {}
    sum_lines: List[str] = []
    for p in built:
        digest = sha256_file(p)
        sums[p.name] = digest
        sum_lines.append(f"{digest}  {p.name}")

    sums_path = DIST_DIR / "SHA256SUMS.txt"
    sums_path.write_text("\n".join(sum_lines) + "\n", encoding="utf-8")

    notes = f"""# Laya Address VN v{VERSION}

Phiên bản đóng gói chính thức gồm 2 bản triển khai độc lập (**Web ONNX WASM** và **Backend CPU INT8**) kèm **Python Wheel** cài trực tiếp và **Bộ công cụ đo trần dữ liệu thật offline**.

## Gói đính kèm trong bản phát hành

| Tệp | Mục đích | Dung lượng |
|---|---|---:|
| `laya-address-vn-way-a-v{VERSION}-web.zip` | **Bản dùng trực tiếp trên Web (Way A v2)** — HTML/JS/CSS + Char-CNN ONNX (`1.283 MB`) + MIT Gazetteer + `onnxruntime-web@1.20.1` (`0` CDN, `0đ` server) | `{(built[0].stat().st_size / (1024 * 1024)):.2f} MB` |
| `laya-address-vn-way-b-v{VERSION}-backend.zip` | **Bản Backend CPU (Way B v2)** — FastAPI server + Dual-Granularity Transformer/Char-CNN ONNX INT8 (`1.789 MB`) + MIT Gazetteer + `requirements.txt` | `{(built[1].stat().st_size / (1024 * 1024)):.2f} MB` |
| `laya_address_vn-{VERSION}-py3-none-any.whl` | **Python Wheel (Way B v2)** — Cài thẳng `pip install` để nhúng `LayaAddressEngine` vào code Python (`+ [server]` nếu chạy FastAPI) | `{(built[2].stat().st_size / (1024 * 1024)):.2f} MB` |
| `laya-address-vn-ceiling-audit-v{VERSION}.zip` | **Bộ kiểm chứng đơn thật offline** — Bước 1 đo trần Pre/Post-Goong + Bước 2 lấy 300 mẫu mù IPW (`0%` PII rò rỉ) | `{(built[3].stat().st_size / 1024):.1f} KB` |
| `SHA256SUMS.txt` | Mã kiểm tra toàn vẹn SHA-256 cho cả 4 gói | — |

## Cài đặt nhanh

### 1. Dùng trực tiếp trên Web (Way A v2)
Giải nén `laya-address-vn-way-a-v{VERSION}-web.zip` và copy thư mục `web/` lên static hosting (hoặc chạy thử `python serve.py` -> `http://127.0.0.1:8083`). Xem hướng dẫn nhúng trong `QUICKSTART_WEB.md`.

### 2. Dùng ở Backend Python / FastAPI (Way B v2)
```bash
# Cách 1: Cài thẳng bằng Python Wheel
pip install laya_address_vn-{VERSION}-py3-none-any.whl
# Hoặc kèm FastAPI server:
pip install "laya_address_vn-{VERSION}-py3-none-any.whl[server]"
uvicorn laya_address_vn.server:app --host 0.0.0.0 --port 8084
```
```python
from laya_address_vn import LayaAddressEngine

engine = LayaAddressEngine()
res = engine.decide("Lộc Thạnh, Lộc Ninh", mode="hybrid")
print(res.is_address, res.sufficiency, res.probability)
```

## Kết quả tự kiểm độc lập (600 Hold-Out `seed=20261122`, chế độ `hybrid`)
- **Way A v2:** `98.33%` 3-class Acc • `0.56%` (`1/180`) chặn nhầm địa chỉ ngắn hợp lệ • `100.00%` recall `insufficient` • `100.00%` chặn rác (`11/11`) • CPU `p99 = 0.381 ms`
- **Way B v2:** `99.83%` 3-class Acc • `0.56%` (`1/180`) chặn nhầm địa chỉ ngắn hợp lệ • `100.00%` recall `insufficient` • `100.00%` chặn rác (`11/11`) • CPU `p99 = 1.097 ms`

## SHA-256 Checksums
```text
{chr(10).join(sum_lines)}
```
"""
    (DIST_DIR / "RELEASE_NOTES.md").write_text(notes, encoding="utf-8")

    for p in built:
        print(f"BUILT {p.name:48s}  {p.stat().st_size:10,d} B  sha256={sums[p.name]}")
    print(f"BUILT SHA256SUMS.txt & RELEASE_NOTES.md in {DIST_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
