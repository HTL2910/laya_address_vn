"""Step 2 Stratified Disagreement Sampler + Offline HTML/Excel Labeling Toolkit.

Partitions the Rule-D-passed residual slice (deduplicated on normalized text + Goong)
into mutually exclusive strata:
  - Stratum 1: Disagreement (Rule D pass, Way A v2 OR Way B v2 block — or top OOF risk)
  - Stratum 2: Rule D pass, not Stratum 1, Returned (split into high Post-Goong risk & other if OOF exists)
  - Stratum 3: Rule D pass, not Stratum 1, Delivered

Outputs strictly blinded labeling materials (`blind_label_sheet_300.csv/.xlsx`,
standalone `label_300_ui.html`) and separate answer/weight key (`sample_300_key.csv`)
with exact Horvitz-Thompson IPW weights summing to the deduplicated Rule-D-pass population.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation
import pandas as pd

from post_goong_features import VietnamAdminIndex, extract_feature_dataframe
from schema_io import (
    load_dataset,
    parse_bool_gate,
    parse_is_return,
    populate_goong_columns,
    resolve_schema,
    strip_accents,
    write_csv_utf8_sig,
)


SUFFICIENCY_CHOICES = ("sufficient", "insufficient", "not_address")
GOONG_CHOICES = ("correct", "wrong", "empty", "unclear")
ERROR_TYPE_CHOICES = (
    "none",
    "missing_ward",
    "ambiguous_ward_wrong_prov",
    "typo_or_no_accent",
    "unstructured_rural",
    "chat_or_garbage",
    "product_or_note",
    "goong_hallucination",
    "non_address_return",
)


def _norm_key_str(s: str) -> str:
    return " ".join(strip_accents(s or "").lower().split())


def allocate_quotas(capacities: Dict[str, int], targets: Dict[str, int]) -> Dict[str, int]:
    """Allocate sample sizes up to target quotas, redistributing any shortfall proportionally."""
    chosen = {k: min(capacities[k], targets[k]) for k in targets}
    total_desired = sum(targets.values())
    total_avail = sum(capacities.values())
    goal = min(total_desired, total_avail)

    remaining = goal - sum(chosen.values())
    while remaining > 0:
        headroom = {k: capacities[k] - chosen[k] for k in targets if capacities[k] - chosen[k] > 0}
        if not headroom:
            break
        # Distribute greedily proportional to remaining headroom
        sum_h = sum(headroom.values())
        added_any = False
        for k in sorted(headroom.keys()):
            if remaining <= 0:
                break
            give = min(headroom[k], max(1, (remaining * headroom[k]) // sum_h))
            chosen[k] += give
            remaining -= give
            added_any = True
        if not added_any:
            break
    return chosen


def write_blind_xlsx(df_blind: pd.DataFrame, path: Path) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Blind_Label_300"
    ws.freeze_panes = "A2"

    header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    header_font = Font(color="FFFFFF", bold=True, size=10)

    cols = list(df_blind.columns)
    ws.append(cols)
    for c_i in range(1, len(cols) + 1):
        cell = ws.cell(row=1, column=c_i)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")

    for row in df_blind.itertuples(index=False, name=None):
        ws.append([None if pd.isna(v) else v for v in row])

    widths = {
        "sample_id": 11,
        "raw_text": 48,
        "goong_ward": 22,
        "goong_district": 22,
        "goong_province": 20,
        "goong_formatted": 40,
        "du_den_xa": 18,
        "goong_dung": 16,
        "loai_loi": 28,
        "ghi_chu": 30,
    }
    for c_i, cname in enumerate(cols, start=1):
        ws.column_dimensions[get_column_letter(c_i)].width = widths.get(cname, 20)

    n_rows = len(df_blind) + 1
    if n_rows >= 2:
        col_idx_map = {c: get_column_letter(i + 1) for i, c in enumerate(cols)}
        validations = [
            ("du_den_xa", f'"{",".join(SUFFICIENCY_CHOICES)}"'),
            ("goong_dung", f'"{",".join(GOONG_CHOICES)}"'),
            ("loai_loi", f'"{",".join(ERROR_TYPE_CHOICES)}"'),
        ]
        for col_name, formula in validations:
            if col_name in col_idx_map:
                ltr = col_idx_map[col_name]
                dv = DataValidation(type="list", formula1=formula, allow_blank=True)
                ws.add_data_validation(dv)
                dv.add(f"{ltr}2:{ltr}{n_rows}")

    ws_guide = wb.create_sheet("Huong_Dan")
    guide_rows = [
        ["Cột gán nhãn", "Giá trị hợp lệ", "Ý nghĩa"],
        ["du_den_xa", "sufficient", "Đủ xác định đến Xã/Phường VÀ có ít nhất 1 cấp cha (Huyện hoặc Tỉnh) tương ứng."],
        ["du_den_xa", "insufficient", "Chỉ có tên Xã đơn lẻ, hoặc chỉ có Huyện/Tỉnh, hoặc mảnh địa chỉ (Ngõ 12, Thôn 3)."],
        ["du_den_xa", "not_address", "Không phải địa chỉ (chat, hỏi giá, tên sản phẩm, sđt rác, ký tự lẻ)."],
        ["goong_dung", "correct", "Kết quả Goong auto-fill ĐÚNG Xã/Huyện/Tỉnh so với raw_text."],
        ["goong_dung", "wrong", "Goong điền SAI xã/huyện/tỉnh (bịa cấp cha cho xã trùng tên, mâu thuẫn tỉnh/huyện)."],
        ["goong_dung", "empty", "Goong không trả ra được Xã/Phường (trống)."],
        ["goong_dung", "unclear", "Không đủ thông tin trong raw_text để kết luận Goong đúng hay sai."],
        ["loai_loi", "none / ...", "Phân loại nguyên nhân gốc để tổng hợp cho v3."],
    ]
    for gr in guide_rows:
        ws_guide.append(gr)
    ws_guide.column_dimensions["A"].width = 18
    ws_guide.column_dimensions["B"].width = 24
    ws_guide.column_dimensions["C"].width = 75

    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def render_offline_html_ui(items: List[Dict[str, str]], storage_key: str) -> str:
    """Build 100% self-contained single-file HTML labeling UI (zero CDN links)."""
    embedded_json = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Laya Address VN — Gán Nhãn Mù 300 Mẫu (Offline)</title>
<style>
  :root {{
    --bg: #0f172a; --card: #1e293b; --muted: #94a3b8; --text: #f1f5f9;
    --border: #334155; --accent: #38bdf8; --ok: #22c55e; --warn: #f59e0b; --bad: #ef4444;
  }}
  * {{ box-sizing: border-box; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }}
  body {{ margin: 0; background: var(--bg); color: var(--text); padding: 18px; }}
  .wrap {{ max-width: 980px; margin: 0 auto; }}
  header {{ display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 14px; }}
  h1 {{ font-size: 18px; margin: 0; color: var(--accent); }}
  .controls {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
  button, select, input[type="text"] {{
    background: var(--card); color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: 7px 11px; font-size: 13px; cursor: pointer;
  }}
  button:hover {{ border-color: var(--accent); }}
  button.primary {{ background: #0284c7; border-color: #0ea5e9; font-weight: 600; }}
  button.active {{ outline: 2px solid var(--accent); background: #0c4a6e; }}
  button.choice-ok.active {{ background: #14532d; outline-color: var(--ok); }}
  button.choice-warn.active {{ background: #78350f; outline-color: var(--warn); }}
  button.choice-bad.active {{ background: #7f1d1d; outline-color: var(--bad); }}
  .prog-bar {{ height: 8px; background: #1e293b; border-radius: 99px; overflow: hidden; margin-bottom: 14px; border: 1px solid var(--border); }}
  .prog-fill {{ height: 100%; background: var(--ok); width: 0%; transition: width 0.2s; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 16px; margin-bottom: 14px; }}
  .meta-row {{ display: flex; justify-content: space-between; color: var(--muted); font-size: 13px; margin-bottom: 10px; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
  @media (max-width: 720px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
  .box {{ background: #0f172a; border: 1px solid var(--border); border-radius: 8px; padding: 12px; }}
  .box-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin-bottom: 6px; }}
  .raw-text {{ font-size: 18px; line-height: 1.45; font-weight: 600; word-break: break-word; }}
  .goong-kv {{ font-size: 14px; line-height: 1.65; }}
  .goong-kv b {{ color: var(--accent); }}
  .section-title {{ font-size: 13px; color: var(--muted); margin: 12px 0 6px; font-weight: 600; }}
  .btn-group {{ display: flex; flex-wrap: wrap; gap: 8px; }}
  .shortcut {{ font-size: 11px; color: var(--muted); margin-left: 4px; }}
  input[type="text"] {{ width: 100%; cursor: text; margin-top: 6px; }}
  .footer-nav {{ display: flex; justify-content: space-between; align-items: center; margin-top: 12px; flex-wrap: wrap; gap: 10px; }}
  .hint {{ color: var(--muted); font-size: 12px; margin-top: 10px; }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>Laya Address VN — Gán Nhãn Mù 300 Mẫu (100% Offline)</h1>
      <div style="font-size:12px;color:var(--muted)" id="stats-text">Đã gán: 0 / 0</div>
    </div>
    <div class="controls">
      <select id="filter-mode">
        <option value="all">Tất cả mẫu</option>
        <option value="unlabeled">Chỉ mẫu chưa gán đủ 2 nhãn</option>
      </select>
      <label style="font-size:12px;color:var(--muted)"><input type="checkbox" id="auto-next" /> Tự chuyển câu</label>
      <button id="btn-import">Nạp CSV làm tiếp</button>
      <input type="file" id="file-import" accept=".csv" style="display:none" />
      <button class="primary" id="btn-export">Xuất labeled_300.csv</button>
    </div>
  </header>

  <div class="prog-bar"><div class="prog-fill" id="prog-fill"></div></div>

  <div class="card">
    <div class="meta-row">
      <span>Mã mẫu: <b id="sample-id" style="color:var(--text)">-</b></span>
      <span id="pos-label">0 / 0</span>
    </div>
    <div class="grid-2">
      <div class="box">
        <div class="box-label">Chuỗi khách nhập (raw_text)</div>
        <div class="raw-text" id="raw-text">-</div>
      </div>
      <div class="box">
        <div class="box-label">Kết quả Goong tự điền (Blind View)</div>
        <div class="goong-kv">
          <div><b>Xã/Phường:</b> <span id="g-ward">-</span></div>
          <div><b>Quận/Huyện:</b> <span id="g-dist">-</span></div>
          <div><b>Tỉnh/TP:</b> <span id="g-prov">-</span></div>
          <div style="font-size:12px;color:var(--muted);margin-top:4px" id="g-fmt"></div>
        </div>
      </div>
    </div>

    <div class="section-title">1. Địa chỉ có đủ để xác định đến Xã/Phường không? (du_den_xa)</div>
    <div class="btn-group" id="grp-suf">
      <button data-val="sufficient" class="choice-ok">Đủ đến xã <span class="shortcut">[1]</span></button>
      <button data-val="insufficient" class="choice-warn">Chưa đủ đến xã <span class="shortcut">[2]</span></button>
      <button data-val="not_address" class="choice-bad">Không phải địa chỉ <span class="shortcut">[3]</span></button>
    </div>

    <div class="section-title">2. Goong điền Xã/Huyện/Tỉnh có đúng với raw_text không? (goong_dung)</div>
    <div class="btn-group" id="grp-goong">
      <button data-val="correct" class="choice-ok">Goong đúng <span class="shortcut">[Q]</span></button>
      <button data-val="wrong" class="choice-bad">Goong sai / bịa <span class="shortcut">[W]</span></button>
      <button data-val="empty" class="choice-warn">Goong trống <span class="shortcut">[E]</span></button>
      <button data-val="unclear">Không rõ <span class="shortcut">[R]</span></button>
    </div>

    <div class="section-title">3. Phân loại lỗi gốc (loai_loi — tuỳ chọn)</div>
    <div class="btn-group" id="grp-err">
      <button data-val="none">none (không lỗi)</button>
      <button data-val="missing_ward">missing_ward</button>
      <button data-val="ambiguous_ward_wrong_prov">ambiguous_ward_wrong_prov</button>
      <button data-val="typo_or_no_accent">typo_or_no_accent</button>
      <button data-val="unstructured_rural">unstructured_rural</button>
      <button data-val="chat_or_garbage">chat_or_garbage</button>
      <button data-val="product_or_note">product_or_note</button>
      <button data-val="goong_hallucination">goong_hallucination</button>
      <button data-val="non_address_return">non_address_return</button>
    </div>

    <input type="text" id="note-input" placeholder="Ghi chú ngắn (ghi_chu — tự lưu vào trình duyệt)..." />

    <div class="footer-nav">
      <div class="btn-group">
        <button id="btn-prev">← Câu trước</button>
        <button id="btn-next">Câu tiếp →</button>
      </div>
      <div class="btn-group">
        <button id="btn-clear-cur">Xóa nhãn câu này</button>
      </div>
    </div>
    <div class="hint">Phím tắt: <b>1 / 2 / 3</b> = Đủ đến xã | <b>Q / W / E / R</b> = Goong đúng/sai | <b>← / →</b> = Chuyển câu. Tự lưu local, không gửi đi đâu.</div>
  </div>
</div>

<script>
const ITEMS = {embedded_json};
const STORAGE_KEY = {json.dumps(storage_key)};
let labels = {{}};
try {{
  labels = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{{}}");
}} catch (_) {{ labels = {{}}; }}

let filteredIdx = ITEMS.map((_, i) => i);
let cursor = 0;

function saveStore() {{
  try {{ localStorage.setItem(STORAGE_KEY, JSON.stringify(labels)); }} catch (_) {{}}
}}

function isItemDone(sid) {{
  const l = labels[sid];
  return Boolean(l && l.du_den_xa && l.goong_dung);
}}

function rebuildFilter(keepSid) {{
  const mode = document.getElementById("filter-mode").value;
  filteredIdx = [];
  ITEMS.forEach((it, idx) => {{
    if (mode === "all" || !isItemDone(it.sample_id)) filteredIdx.push(idx);
  }});
  if (filteredIdx.length === 0) filteredIdx = ITEMS.map((_, i) => i);
  if (keepSid) {{
    const p = filteredIdx.findIndex(i => ITEMS[i].sample_id === keepSid);
    cursor = p >= 0 ? p : 0;
  }} else if (cursor >= filteredIdx.length) {{
    cursor = Math.max(0, filteredIdx.length - 1);
  }}
}}

function updateView() {{
  const doneCount = ITEMS.filter(it => isItemDone(it.sample_id)).length;
  document.getElementById("stats-text").textContent = `Đã gán đủ 2 nhãn: ${{doneCount}} / ${{ITEMS.length}}`;
  document.getElementById("prog-fill").style.width = `${{(doneCount / Math.max(1, ITEMS.length)) * 100}}%`;

  const realIdx = filteredIdx[cursor] ?? 0;
  const item = ITEMS[realIdx];
  const lbl = labels[item.sample_id] || {{ du_den_xa: "", goong_dung: "", loai_loi: "", ghi_chu: "" }};

  document.getElementById("sample-id").textContent = item.sample_id;
  document.getElementById("pos-label").textContent = `Vị trí ${{cursor + 1}} / ${{filteredIdx.length}}`;
  document.getElementById("raw-text").textContent = item.raw_text || "(trống)";
  document.getElementById("g-ward").textContent = item.goong_ward || "(trống)";
  document.getElementById("g-dist").textContent = item.goong_district || "(trống)";
  document.getElementById("g-prov").textContent = item.goong_province || "(trống)";
  document.getElementById("g-fmt").textContent = item.goong_formatted ? ("Goong full: " + item.goong_formatted) : "";
  document.getElementById("note-input").value = lbl.ghi_chu || "";

  document.querySelectorAll("#grp-suf button").forEach(b => b.classList.toggle("active", b.dataset.val === lbl.du_den_xa));
  document.querySelectorAll("#grp-goong button").forEach(b => b.classList.toggle("active", b.dataset.val === lbl.goong_dung));
  document.querySelectorAll("#grp-err button").forEach(b => b.classList.toggle("active", b.dataset.val === lbl.loai_loi));
}}

function setField(field, val) {{
  const item = ITEMS[filteredIdx[cursor] ?? 0];
  const cur = labels[item.sample_id] || {{ du_den_xa: "", goong_dung: "", loai_loi: "", ghi_chu: "" }};
  cur[field] = val;
  if (field === "du_den_xa" && val === "sufficient" && cur.goong_dung === "correct" && !cur.loai_loi) {{
    cur.loai_loi = "none";
  }}
  if (field === "goong_dung" && val === "correct" && cur.du_den_xa === "sufficient" && !cur.loai_loi) {{
    cur.loai_loi = "none";
  }}
  labels[item.sample_id] = cur;
  saveStore();
  if (document.getElementById("auto-next").checked && isItemDone(item.sample_id) && cursor < filteredIdx.length - 1) {{
    cursor += 1;
  }}
  updateView();
}}

document.querySelectorAll("#grp-suf button").forEach(b => b.addEventListener("click", () => setField("du_den_xa", b.dataset.val)));
document.querySelectorAll("#grp-goong button").forEach(b => b.addEventListener("click", () => setField("goong_dung", b.dataset.val)));
document.querySelectorAll("#grp-err button").forEach(b => b.addEventListener("click", () => setField("loai_loi", b.dataset.val)));

document.getElementById("note-input").addEventListener("input", (e) => {{
  const item = ITEMS[filteredIdx[cursor] ?? 0];
  const cur = labels[item.sample_id] || {{ du_den_xa: "", goong_dung: "", loai_loi: "", ghi_chu: "" }};
  cur.ghi_chu = e.target.value;
  labels[item.sample_id] = cur;
  saveStore();
}});

document.getElementById("btn-prev").addEventListener("click", () => {{ if (cursor > 0) {{ cursor -= 1; updateView(); }} }});
document.getElementById("btn-next").addEventListener("click", () => {{ if (cursor < filteredIdx.length - 1) {{ cursor += 1; updateView(); }} }});
document.getElementById("btn-clear-cur").addEventListener("click", () => {{
  const item = ITEMS[filteredIdx[cursor] ?? 0];
  delete labels[item.sample_id];
  saveStore();
  updateView();
}});
document.getElementById("filter-mode").addEventListener("change", () => {{
  const curSid = ITEMS[filteredIdx[cursor] ?? 0]?.sample_id;
  rebuildFilter(curSid);
  updateView();
}});

function csvCell(v) {{
  const s = String(v ?? "");
  return /[",\\r\\n]/.test(s) ? `"${{s.replace(/"/g, '""')}}"` : s;
}}

document.getElementById("btn-export").addEventListener("click", () => {{
  const header = ["sample_id","raw_text","goong_ward","goong_district","goong_province","goong_formatted","du_den_xa","goong_dung","loai_loi","ghi_chu"];
  const lines = [header.join(",")];
  ITEMS.forEach(it => {{
    const l = labels[it.sample_id] || {{}};
    lines.push([
      it.sample_id, it.raw_text, it.goong_ward, it.goong_district, it.goong_province, it.goong_formatted,
      l.du_den_xa || "", l.goong_dung || "", l.loai_loi || "", l.ghi_chu || ""
    ].map(csvCell).join(","));
  }});
  const blob = new Blob(["\\uFEFF" + lines.join("\\r\\n")], {{ type: "text/csv;charset=utf-8;" }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "labeled_300.csv";
  a.click();
  URL.revokeObjectURL(url);
}});

function parseCSV(text) {{
  const rows = [];
  let row = [], val = "", inQ = false;
  for (let i = 0; i < text.length; i++) {{
    const ch = text[i];
    if (inQ) {{
      if (ch === '"' && text[i+1] === '"') {{ val += '"'; i++; }}
      else if (ch === '"') inQ = false;
      else val += ch;
    }} else {{
      if (ch === '"') inQ = true;
      else if (ch === ",") {{ row.push(val); val = ""; }}
      else if (ch === "\\n") {{ row.push(val.replace(/\\r$/, "")); rows.push(row); row = []; val = ""; }}
      else val += ch;
    }}
  }}
  if (val.length > 0 || row.length > 0) {{ row.push(val.replace(/\\r$/, "")); rows.push(row); }}
  return rows;
}}

document.getElementById("btn-import").addEventListener("click", () => document.getElementById("file-import").click());
document.getElementById("file-import").addEventListener("change", (ev) => {{
  const file = ev.target.files?.[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {{
    const clean = String(reader.result || "").replace(/^\\uFEFF/, "");
    const rows = parseCSV(clean);
    if (rows.length < 2) return;
    const hdr = rows[0].map(h => h.trim());
    const idxSid = hdr.indexOf("sample_id");
    const idxSuf = hdr.indexOf("du_den_xa");
    const idxGoong = hdr.indexOf("goong_dung");
    const idxErr = hdr.indexOf("loai_loi");
    const idxNote = hdr.indexOf("ghi_chu");
    if (idxSid < 0) return;
    for (let r = 1; r < rows.length; r++) {{
      const cols = rows[r];
      const sid = (cols[idxSid] || "").trim();
      if (!sid) continue;
      labels[sid] = {{
        du_den_xa: idxSuf >= 0 ? (cols[idxSuf] || "").trim() : "",
        goong_dung: idxGoong >= 0 ? (cols[idxGoong] || "").trim() : "",
        loai_loi: idxErr >= 0 ? (cols[idxErr] || "").trim() : "",
        ghi_chu: idxNote >= 0 ? (cols[idxNote] || "").trim() : "",
      }};
    }}
    saveStore();
    rebuildFilter();
    updateView();
  }};
  reader.readAsText(file, "utf-8");
}});

document.addEventListener("keydown", (e) => {{
  if (document.activeElement && document.activeElement.tagName === "INPUT") return;
  if (e.key === "1") setField("du_den_xa", "sufficient");
  else if (e.key === "2") setField("du_den_xa", "insufficient");
  else if (e.key === "3") setField("du_den_xa", "not_address");
  else if (e.key.toLowerCase() === "q") setField("goong_dung", "correct");
  else if (e.key.toLowerCase() === "w") setField("goong_dung", "wrong");
  else if (e.key.toLowerCase() === "e") setField("goong_dung", "empty");
  else if (e.key.toLowerCase() === "r") setField("goong_dung", "unclear");
  else if (e.key === "ArrowLeft" && cursor > 0) {{ cursor -= 1; updateView(); }}
  else if (e.key === "ArrowRight" && cursor < filteredIdx.length - 1) {{ cursor += 1; updateView(); }}
}});

rebuildFilter();
updateView();
</script>
</body>
</html>
"""


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Laya Address VN — Step 2 Stratified 300-Sample Builder")
    p.add_argument("--data", required=True, help="Đường dẫn file data DEV")
    p.add_argument("--sheet", default=None)
    p.add_argument("--oof-csv", default=None, help="File oof_predictions.csv từ Bước 1 (nếu có)")
    p.add_argument("--output-dir", default=None)

    p.add_argument("--n-disagree", type=int, default=100)
    p.add_argument("--n-return", type=int, default=100)
    p.add_argument("--n-deliver", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--score-v2", action="store_true", help="Chạy Way A/B v2 trực tiếp nếu CSV chưa có sẵn")

    p.add_argument("--order-id-col", default=None)
    p.add_argument("--text-col", default=None)
    p.add_argument("--outcome-col", default=None)
    p.add_argument("--rule-d-col", default=None)
    p.add_argument("--rule-d-is-block", action="store_true")
    p.add_argument("--goong-ward-col", default=None)
    p.add_argument("--goong-district-col", default=None)
    p.add_argument("--goong-province-col", default=None)
    p.add_argument("--goong-score-col", default=None)
    p.add_argument("--goong-formatted-col", default=None)
    p.add_argument("--way-a-col", default=None)
    p.add_argument("--way-b-col", default=None)
    p.add_argument("--return-values", default=None)
    p.add_argument("--delivered-values", default=None)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    out_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).resolve().parent / "outputs" / f"label_300_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_dataset(args.data, sheet_name=args.sheet)
    schema = resolve_schema(df_raw, overrides={
        "order_id": args.order_id_col,
        "text": args.text_col,
        "outcome": args.outcome_col,
        "rule_d": args.rule_d_col,
        "rule_d_is_block": args.rule_d_is_block,
        "goong_ward": args.goong_ward_col,
        "goong_district": args.goong_district_col,
        "goong_province": args.goong_province_col,
        "goong_score": args.goong_score_col,
        "goong_formatted": args.goong_formatted_col,
        "way_a": args.way_a_col,
        "way_b": args.way_b_col,
    })

    ret_vals = [v.strip() for v in args.return_values.split(",")] if args.return_values else None
    del_vals = [v.strip() for v in args.delivered_values.split(",")] if args.delivered_values else None
    y_all = parse_is_return(df_raw[schema.outcome], return_values=ret_vals, delivered_values=del_vals)
    valid_mask = (y_all == 0) | (y_all == 1)
    df_valid = df_raw.loc[valid_mask].copy().reset_index(drop=True)
    y_valid = y_all[valid_mask].astype(np.int64)

    if schema.order_id and schema.order_id in df_valid.columns:
        df_valid["_order_id"] = df_valid[schema.order_id].astype(str)
    else:
        df_valid["_order_id"] = [f"ROW_{i + 1:06d}" for i in range(len(df_valid))]

    if schema.rule_d and schema.rule_d in df_valid.columns:
        rule_d_pass = parse_bool_gate(df_valid[schema.rule_d], invert=schema.rule_d_is_block)
    else:
        rule_d_pass = np.ones(len(df_valid), dtype=bool)

    df_res = df_valid.loc[rule_d_pass].copy().reset_index(drop=True)
    y_res = y_valid[rule_d_pass]
    n_raw_residual = len(df_res)
    if n_raw_residual == 0:
        raise ValueError("Không có đơn nào lọt qua Luật D để lấy mẫu Bước 2.")

    raw_texts = [
        "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip()
        for v in df_res[schema.text].tolist()
    ]
    g_wards, g_dists, g_provs, _ = populate_goong_columns(df_res, schema)
    g_fmts = (
        ["" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip() for v in df_res[schema.goong_formatted].tolist()]
        if (schema.goong_formatted and schema.goong_formatted in df_res.columns)
        else [""] * n_raw_residual
    )

    df_res["_raw_text"] = raw_texts
    df_res["_is_return"] = y_res
    df_res["_goong_ward"] = g_wards
    df_res["_goong_district"] = g_dists
    df_res["_goong_province"] = g_provs
    df_res["_goong_formatted"] = g_fmts

    # Attach Way A / Way B flags if present
    has_way_cols = False
    if schema.way_a and schema.way_a in df_res.columns:
        df_res["_way_a_pass"] = parse_bool_gate(df_res[schema.way_a])
        has_way_cols = True
    else:
        df_res["_way_a_pass"] = np.nan

    if schema.way_b and schema.way_b in df_res.columns:
        df_res["_way_b_pass"] = parse_bool_gate(df_res[schema.way_b])
        has_way_cols = True
    else:
        df_res["_way_b_pass"] = np.nan

    # Merge OOF predictions from Step 1 if provided
    df_res["_oof_pre_goong_text"] = np.nan
    df_res["_oof_pre_goong_struct"] = np.nan
    df_res["_oof_post_goong_consistency"] = np.nan
    df_res["_oof_post_goong_combined"] = np.nan
    df_res["_diagnostic_bucket"] = "unknown"

    if args.oof_csv:
        oof_df = load_dataset(args.oof_csv)
        if len(oof_df) == n_raw_residual and "_order_id" in df_res.columns and "order_id" in oof_df.columns:
            # Align carefully by order_id if 1-1 unique, else positional
            if df_res["_order_id"].is_unique and oof_df["order_id"].astype(str).is_unique:
                oof_idx = oof_df.set_index(oof_df["order_id"].astype(str)).reindex(df_res["_order_id"]).reset_index(drop=True)
            else:
                oof_idx = oof_df.reset_index(drop=True)
            for src, dst in (
                ("oof_pre_goong_text", "_oof_pre_goong_text"),
                ("oof_pre_goong_struct", "_oof_pre_goong_struct"),
                ("oof_post_goong_consistency", "_oof_post_goong_consistency"),
                ("oof_post_goong_combined", "_oof_post_goong_combined"),
            ):
                if src in oof_idx.columns:
                    df_res[dst] = pd.to_numeric(oof_idx[src], errors="coerce")
            if "diagnostic_bucket" in oof_idx.columns:
                df_res["_diagnostic_bucket"] = oof_idx["diagnostic_bucket"].fillna("unknown").astype(str)
            if not has_way_cols:
                if "way_a_v2_pass" in oof_idx.columns and oof_idx["way_a_v2_pass"].notna().any():
                    df_res["_way_a_pass"] = parse_bool_gate(oof_idx["way_a_v2_pass"])
                    has_way_cols = True
                if "way_b_v2_pass" in oof_idx.columns and oof_idx["way_b_v2_pass"].notna().any():
                    df_res["_way_b_pass"] = parse_bool_gate(oof_idx["way_b_v2_pass"])
                    has_way_cols = True

    if args.score_v2 and not has_way_cols:
        from v2_scorer import score_way_a_v2_batch, score_way_b_v2_batch

        sa = score_way_a_v2_batch(raw_texts, mode="hybrid")
        if sa is not None:
            df_res["_way_a_pass"] = np.asarray(sa["pass"], dtype=bool)
            has_way_cols = True
        sb = score_way_b_v2_batch(raw_texts, mode="hybrid")
        df_res["_way_b_pass"] = np.asarray(sb["pass"], dtype=bool)
        has_way_cols = True

    if (df_res["_diagnostic_bucket"] == "unknown").all():
        index = VietnamAdminIndex()
        _, _, buckets = extract_feature_dataframe(df_res, schema, index=index)
        df_res["_diagnostic_bucket"] = buckets

    # Deduplicate cleanly on normalized (raw_text, goong_ward, goong_district, goong_province)
    df_res["_dedup_key"] = [
        (_norm_key_str(t), _norm_key_str(w), _norm_key_str(d), _norm_key_str(p))
        for t, w, d, p in zip(raw_texts, g_wards, g_dists, g_provs)
    ]
    # Shuffle before drop_duplicates so representative outcome isn't ordered by time/label
    rng = np.random.default_rng(args.seed)
    shuf_order = rng.permutation(n_raw_residual)
    df_shuf = df_res.iloc[shuf_order].copy().reset_index(drop=True)
    dup_counts = df_shuf["_dedup_key"].value_counts()
    df_uniq = df_shuf.drop_duplicates(subset=["_dedup_key"], keep="first").copy().reset_index(drop=True)
    df_uniq["_dup_count"] = df_uniq["_dedup_key"].map(dup_counts).astype(int)
    n_pop_dedup = len(df_uniq)

    # Assign mutually exclusive strata partitioning df_uniq (N_pop = n_pop_dedup)
    if has_way_cols:
        wa_blk = df_uniq["_way_a_pass"].map(lambda v: False if pd.isna(v) else (not bool(v))).to_numpy(dtype=bool)
        wb_blk = df_uniq["_way_b_pass"].map(lambda v: False if pd.isna(v) else (not bool(v))).to_numpy(dtype=bool)
        is_disagree = wa_blk | wb_blk
        disagree_name = "disagreement_v2_block"
    elif df_uniq["_oof_post_goong_consistency"].notna().any() or df_uniq["_oof_pre_goong_text"].notna().any():
        sc_col = "_oof_post_goong_consistency" if df_uniq["_oof_post_goong_consistency"].notna().any() else "_oof_pre_goong_text"
        sc_vals = df_uniq[sc_col].fillna(-1.0).to_numpy(dtype=np.float64)
        thr95 = float(np.nanpercentile(sc_vals, 95))
        is_disagree = sc_vals >= thr95
        disagree_name = "disagreement_ceiling_top_risk"
    else:
        is_disagree = (df_uniq["_diagnostic_bucket"] != "grounded_consistent").to_numpy(dtype=bool)
        disagree_name = "disagreement_heuristic_risk"

    strata = np.empty(n_pop_dedup, dtype=object)
    y_u = df_uniq["_is_return"].to_numpy(dtype=np.int64)
    has_post_oof = bool(df_uniq["_oof_post_goong_consistency"].notna().sum() >= 20)

    for i in range(n_pop_dedup):
        if is_disagree[i]:
            strata[i] = disagree_name
        elif y_u[i] == 1:
            strata[i] = "rule_d_pass_returned"
        else:
            strata[i] = "rule_d_pass_delivered"

    # Optionally sub-stratify non-disagree returned stratum cleanly if Post-Goong OOF risk exists
    ret_idx_all = np.flatnonzero(strata == "rule_d_pass_returned")
    if has_post_oof and len(ret_idx_all) >= 40:
        ret_scores = df_uniq["_oof_post_goong_consistency"].iloc[ret_idx_all].to_numpy(dtype=np.float64)
        med = float(np.nanmedian(ret_scores))
        high_m = ret_scores >= med
        # Only apply split if both halves contain meaningful observations
        if int(np.sum(high_m)) >= 15 and int(np.sum(~high_m)) >= 15:
            strata[ret_idx_all[high_m]] = "rule_d_pass_returned_high_post_goong_risk"
            strata[ret_idx_all[~high_m]] = "rule_d_pass_returned_other"

    df_uniq["_stratum"] = strata
    strata_counts = pd.Series(strata).value_counts().to_dict()

    if "rule_d_pass_returned_high_post_goong_risk" in strata_counts:
        targets = {
            disagree_name: args.n_disagree,
            "rule_d_pass_returned_high_post_goong_risk": args.n_return // 2,
            "rule_d_pass_returned_other": args.n_return - (args.n_return // 2),
            "rule_d_pass_delivered": args.n_deliver,
        }
    else:
        targets = {
            disagree_name: args.n_disagree,
            "rule_d_pass_returned": args.n_return,
            "rule_d_pass_delivered": args.n_deliver,
        }

    capacities = {k: int(strata_counts.get(k, 0)) for k in targets}
    quotas = allocate_quotas(capacities, targets)

    sampled_frames: List[pd.DataFrame] = []
    for k, q in quotas.items():
        if q <= 0:
            continue
        sub = df_uniq.loc[df_uniq["_stratum"] == k].copy()
        n_tot_k = len(sub)
        picked_pos = rng.choice(n_tot_k, size=q, replace=False)
        sub_s = sub.iloc[picked_pos].copy()
        sub_s["_stratum_pop_n"] = n_tot_k
        sub_s["_stratum_sample_n"] = q
        sub_s["_stratum_weight"] = float(n_tot_k / q)
        sampled_frames.append(sub_s)

    df_sampled = pd.concat(sampled_frames, ignore_index=True)
    # Shuffle final 300 rows so labeler cannot infer stratum from row order
    final_perm = rng.permutation(len(df_sampled))
    df_sampled = df_sampled.iloc[final_perm].copy().reset_index(drop=True)
    df_sampled["sample_id"] = [f"S{i + 1:03d}" for i in range(len(df_sampled))]

    # Exact Horvitz-Thompson population invariant check
    weight_sum = float(df_sampled["_stratum_weight"].sum())
    assert abs(weight_sum - n_pop_dedup) < 1e-6, (
        f"IPW weight sum mismatch: sum={weight_sum} vs N_pop={n_pop_dedup}"
    )

    # 1. Blind CSV & XLSX (ZERO outcome / stratum / model scores)
    df_blind = pd.DataFrame({
        "sample_id": df_sampled["sample_id"],
        "raw_text": df_sampled["_raw_text"],
        "goong_ward": df_sampled["_goong_ward"],
        "goong_district": df_sampled["_goong_district"],
        "goong_province": df_sampled["_goong_province"],
        "goong_formatted": df_sampled["_goong_formatted"],
        "du_den_xa": "",
        "goong_dung": "",
        "loai_loi": "",
        "ghi_chu": "",
    })
    write_csv_utf8_sig(df_blind, out_dir / "blind_label_sheet_300.csv")
    write_blind_xlsx(df_blind, out_dir / "blind_label_sheet_300.xlsx")

    # 2. Separate Metadata/Answer Key (ZERO raw customer text!)
    df_key = pd.DataFrame({
        "sample_id": df_sampled["sample_id"],
        "order_id": df_sampled["_order_id"],
        "stratum": df_sampled["_stratum"],
        "stratum_pop_n": df_sampled["_stratum_pop_n"],
        "stratum_sample_n": df_sampled["_stratum_sample_n"],
        "stratum_weight": df_sampled["_stratum_weight"],
        "dup_count": df_sampled["_dup_count"],
        "is_return": df_sampled["_is_return"],
        "rule_d_pass": True,
        "way_a_v2_pass": df_sampled["_way_a_pass"],
        "way_b_v2_pass": df_sampled["_way_b_pass"],
        "diagnostic_bucket": df_sampled["_diagnostic_bucket"],
        "oof_pre_goong_text": df_sampled["_oof_pre_goong_text"],
        "oof_pre_goong_struct": df_sampled["_oof_pre_goong_struct"],
        "oof_post_goong_consistency": df_sampled["_oof_post_goong_consistency"],
        "oof_post_goong_combined": df_sampled["_oof_post_goong_combined"],
        "n_pop_dedup": n_pop_dedup,
        "n_raw_residual": n_raw_residual,
    })
    write_csv_utf8_sig(df_key, out_dir / "sample_300_key.csv")

    # 3. Standalone Offline HTML Labeling UI
    html_items = [
        {
            "sample_id": str(r["sample_id"]),
            "raw_text": str(r["raw_text"]),
            "goong_ward": str(r["goong_ward"]),
            "goong_district": str(r["goong_district"]),
            "goong_province": str(r["goong_province"]),
            "goong_formatted": str(r["goong_formatted"]),
        }
        for _, r in df_blind.iterrows()
    ]
    storage_key = f"laya_label300_{args.seed}_{len(html_items)}_{n_pop_dedup}"
    (out_dir / "label_300_ui.html").write_text(render_offline_html_ui(html_items, storage_key), encoding="utf-8")

    manifest = {
        "n_raw_residual": int(n_raw_residual),
        "n_pop_dedup": int(n_pop_dedup),
        "dedup_removed": int(n_raw_residual - n_pop_dedup),
        "sampled_total": int(len(df_sampled)),
        "ipw_weight_sum": round(weight_sum, 6),
        "strata_population_counts": {str(k): int(v) for k, v in strata_counts.items()},
        "strata_sampled_counts": {str(k): int(v) for k, v in quotas.items()},
        "has_way_v2_columns": bool(has_way_cols),
    }
    (out_dir / "sampling_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[DONE] Đã lấy {len(df_sampled)} mẫu mù vào: {out_dir}")
    print(f"  -> Mở file HTML offline bằng Chrome/Edge để gán nhãn: {out_dir / 'label_300_ui.html'}")
    print(f"  -> Hoặc gán nhãn trực tiếp trên Excel: {out_dir / 'blind_label_sheet_300.xlsx'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
