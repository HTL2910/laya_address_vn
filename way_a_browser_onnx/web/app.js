/**
 * Tier 2 Browser ONNX WASM Runtime + Hybrid 2-Tier Orchestrator + UI Controller
 * Runs 100% client-side using onnxruntime-web (0đ server).
 */
(function () {
  "use strict";

  const DEFAULT_BATCH_SAMPLES = [
    "Số 123 Đường Nguyễn Trãi, Phường Thượng Đình, Quận Thanh Xuân, Hà Nội",
    "45/12B Cách Mạng Tháng 8, Phường 13, Quận 10, TP. Hồ Chí Minh",
    "123 nguyen trai thanh xuan ha noi",
    "P.12 Q.TB HCM",
    "sn 45 ngo 102 truong chinh hn",
    "15/2B CMT8 P13 Q10",
    "45 Lê Lợi, Q1, HCM",
    "ngõ 68 cầu giấy",
    "giao anh hung 25 tran hung dao hoan kiem",
    "Nguyễn Văn Hùng",
    "0912345678",
    "Áo thun nam size XL màu đen",
    "Hà Nội hôm nay mưa to quá",
    "Quận 1 kẹt xe kinh khủng",
  ];

  const state = {
    session: null,
    vocab: null,
    calibration: null,
    onnxReady: false,
    loadError: null,
    loadTimeMs: null,
  };

  // DOM refs
  const els = {
    statusPill: document.getElementById("engine-status"),
    statusText: document.getElementById("engine-status-text"),
    offlineBanner: document.getElementById("offline-banner"),
    metaSize: document.getElementById("meta-size"),
    metaTemp: document.getElementById("meta-temp"),
    metaAcc: document.getElementById("meta-acc"),
    metaEce: document.getElementById("meta-ece"),
    metaLoad: document.getElementById("meta-load"),
    singleForm: document.getElementById("single-form"),
    singleInput: document.getElementById("single-input"),
    singleClear: document.getElementById("single-clear"),
    singleResult: document.getElementById("single-result"),
    presetChips: document.getElementById("preset-chips"),
    batchForm: document.getElementById("batch-form"),
    batchInput: document.getElementById("batch-input"),
    batchLoadSample: document.getElementById("batch-load-sample"),
    batchClear: document.getElementById("batch-clear"),
    batchSummary: document.getElementById("batch-summary"),
    batchTbody: document.getElementById("batch-tbody"),
  };

  function getSelectedMode() {
    const checked = document.querySelector('input[name="inference_mode"]:checked');
    return checked ? checked.value : "hybrid";
  }

  function setEngineStatus(kind, message) {
    els.statusPill.classList.remove("status-loading", "status-ready", "status-error");
    els.statusPill.classList.add(`status-${kind}`);
    els.statusText.textContent = message;
  }

  function showBanner(msg) {
    els.offlineBanner.textContent = msg;
    els.offlineBanner.hidden = false;
  }

  // Exact same normalization & tokenization as Python train_export.py
  function normalizeText(text) {
    return (text || "").normalize("NFC").trim().replace(/\s+/g, " ").toLowerCase();
  }

  function tokenizeToInt64(text) {
    const { char_to_id, unk_id, max_length } = state.vocab;
    const norm = normalizeText(text);
    const ids = new BigInt64Array(max_length);
    const limit = Math.min(norm.length, max_length);
    for (let i = 0; i < limit; i++) {
      const ch = norm[i];
      const id = Object.prototype.hasOwnProperty.call(char_to_id, ch) ? char_to_id[ch] : unk_id;
      ids[i] = BigInt(id);
    }
    return ids;
  }

  function sigmoid(x) {
    if (x >= 0) {
      const z = Math.exp(-x);
      return 1 / (1 + z);
    }
    const z = Math.exp(x);
    return z / (1 + z);
  }

  async function initEngine() {
    const t0 = performance.now();
    try {
      const [vocabResp, calibResp] = await Promise.all([
        fetch("model/vocab.json"),
        fetch("model/calibration.json"),
      ]);
      if (!vocabResp.ok || !calibResp.ok) {
        throw new Error("Không tìm thấy vocab.json hoặc calibration.json trong model/");
      }
      state.vocab = await vocabResp.json();
      state.calibration = await calibResp.json();

      // Populate metadata cards immediately
      const kb = (state.calibration.model_size_bytes / 1024).toFixed(1);
      els.metaSize.textContent = `${kb} KB`;
      els.metaTemp.textContent = `T = ${Number(state.calibration.temperature).toFixed(4)}`;
      const testM = state.calibration.metrics?.test;
      if (testM) {
        els.metaAcc.textContent = `${(testM.accuracy * 100).toFixed(2)}% / ${(testM.f1 * 100).toFixed(2)}%`;
        els.metaEce.textContent = `${(testM.ece * 100).toFixed(2)}%`;
      }

      // Check onnxruntime-web availability (CDN may be blocked if offline)
      if (typeof window.ort === "undefined") {
        throw new Error(
          "Không thể tải onnxruntime-web từ CDN (có thể máy đang offline). Chế độ Heuristic Only vẫn hoạt động bình thường."
        );
      }

      // Single-threaded WASM avoids SharedArrayBuffer COOP/COEP issues on plain localhost
      window.ort.env.wasm.numThreads = 1;
      window.ort.env.wasm.simd = true;

      state.session = await window.ort.InferenceSession.create("model/laya_address_vn.onnx", {
        executionProviders: ["wasm"],
      });

      // Warm up 1 dummy inference so user latency reflects steady-state WASM execution
      const warmData = new BigInt64Array(state.vocab.max_length);
      const warmTensor = new window.ort.Tensor("int64", warmData, [1, state.vocab.max_length]);
      await state.session.run({ input_ids: warmTensor });

      state.loadTimeMs = +(performance.now() - t0).toFixed(0);
      state.onnxReady = true;
      els.metaLoad.textContent = `${state.loadTimeMs} ms`;
      setEngineStatus("ready", `Sẵn sàng • ONNX WASM (${kb} KB)`);
    } catch (err) {
      state.onnxReady = false;
      state.loadError = err instanceof Error ? err.message : String(err);
      els.metaLoad.textContent = "Offline";
      setEngineStatus("error", "ONNX chưa sẵn sàng (dùng Heuristic)");
      showBanner(`Lưu ý ONNX WASM: ${state.loadError}`);
    }
  }

  async function runOnnxSingle(text) {
    if (!state.onnxReady || !state.session) {
      throw new Error(state.loadError || "Mô hình ONNX WASM đang tải hoặc chưa sẵn sàng.");
    }
    const t0 = performance.now();
    const ids = tokenizeToInt64(text);
    const tensor = new window.ort.Tensor("int64", ids, [1, state.vocab.max_length]);
    const outputs = await state.session.run({ input_ids: tensor });
    const rawLogit = Number(outputs.logit.data[0]);
    const temp = Number(state.calibration.temperature || 1.0);
    const threshold = Number(state.calibration.threshold ?? 0.5);
    const prob = sigmoid(rawLogit / temp);
    const isAddress = prob >= threshold;
    const latencyMs = +(performance.now() - t0).toFixed(3);

    return {
      is_address: isAddress,
      probability: prob,
      raw_logit: +rawLogit.toFixed(4),
      temperature: temp,
      decision_label: isAddress ? "ĐỊA CHỈ HỢP LỆ (TRUE)" : "KHÔNG PHẢI ĐỊA CHỈ (FALSE)",
      tier: "Tier 2 (ONNX WASM)",
      latency_ms: latencyMs,
      reason: `Char-CNN Noul • logit=${rawLogit.toFixed(2)}, T=${temp.toFixed(3)}`,
    };
  }

  async function classify(text, mode) {
    const H = window.LayaHeuristics;
    if (mode === "heuristic") {
      const r = H.evaluateHeuristicOnly(text);
      return {
        is_address: r.is_address,
        probability: r.probability,
        raw_logit: null,
        decision_label: r.is_address ? "ĐỊA CHỈ HỢP LỆ (TRUE)" : "KHÔNG PHẢI ĐỊA CHỈ (FALSE)",
        tier: r.tier,
        latency_ms: r.latency_ms,
        reason: r.reason,
      };
    }

    if (mode === "onnx") {
      if (!state.onnxReady) {
        const fb = H.evaluateHeuristicOnly(text);
        return {
          ...fb,
          decision_label: fb.is_address ? "ĐỊA CHỈ HỢP LỆ (TRUE)" : "KHÔNG PHẢI ĐỊA CHỈ (FALSE)",
          tier: "Fallback Heuristic (ONNX Offline)",
        };
      }
      return await runOnnxSingle(text);
    }

    // Default: Hybrid 2-Tier
    const t1 = H.evaluateTier1(text);
    if (t1.decision !== "ambiguous") {
      return {
        is_address: t1.is_address,
        probability: t1.probability,
        raw_logit: null,
        decision_label: t1.is_address ? "ĐỊA CHỈ HỢP LỆ (TRUE)" : "KHÔNG PHẢI ĐỊA CHỈ (FALSE)",
        tier: "Tier 1 (Heuristic)",
        latency_ms: t1.latency_ms,
        reason: t1.reason,
      };
    }

    if (!state.onnxReady) {
      const fb = H.evaluateHeuristicOnly(text);
      return {
        is_address: fb.is_address,
        probability: fb.probability,
        raw_logit: null,
        decision_label: fb.is_address ? "ĐỊA CHỈ HỢP LỆ (TRUE)" : "KHÔNG PHẢI ĐỊA CHỈ (FALSE)",
        tier: "Tier 1 Fallback (ONNX Offline)",
        latency_ms: +(t1.latency_ms + fb.latency_ms).toFixed(3),
        reason: fb.reason,
      };
    }

    const t2 = await runOnnxSingle(text);
    return {
      ...t2,
      tier: "Tier 2 (ONNX WASM)",
      latency_ms: +(t1.latency_ms + t2.latency_ms).toFixed(3),
      reason: `Tier 1 chuyển tiếp → ${t2.reason}`,
    };
  }

  function renderSingleResult(result) {
    const box = els.singleResult;
    box.classList.remove("result-empty", "result-pos", "result-neg");
    box.classList.add(result.is_address ? "result-pos" : "result-neg");
    box.replaceChildren();

    const pct = (result.probability * 100).toFixed(2);

    const top = document.createElement("div");
    top.className = "result-top";

    const badge = document.createElement("span");
    badge.className = `verdict-badge ${result.is_address ? "verdict-pos" : "verdict-neg"}`;
    badge.textContent = `${result.is_address ? "✓" : "✕"} ${result.decision_label}`;

    const probEl = document.createElement("span");
    probEl.className = "prob-display";
    probEl.textContent = `${pct}%`;

    top.append(badge, probEl);

    const track = document.createElement("div");
    track.className = "prob-track";
    track.setAttribute("aria-hidden", "true");
    const fill = document.createElement("div");
    fill.className = `prob-fill ${result.is_address ? "prob-fill-pos" : "prob-fill-neg"}`;
    fill.style.inlineSize = `${Math.max(2, Math.min(100, result.probability * 100))}%`;
    track.append(fill);

    const meta = document.createElement("div");
    meta.className = "result-meta";

    const addMeta = (label, value) => {
      const span = document.createElement("span");
      span.textContent = `${label}: `;
      const strong = document.createElement("strong");
      strong.textContent = value;
      span.append(strong);
      meta.append(span);
    };

    addMeta("Tầng xử lý", result.tier);
    addMeta("Độ trễ", `${result.latency_ms.toFixed(2)} ms`);
    addMeta("Chi tiết", result.reason);

    box.append(top, track, meta);
  }

  async function handleSingleRun() {
    const text = els.singleInput.value;
    const mode = getSelectedMode();
    const res = await classify(text, mode);
    renderSingleResult(res);
  }

  async function handleBatchRun() {
    const lines = els.batchInput.value
      .split(/\r?\n/)
      .map((l) => l.trim())
      .filter((l) => l.length > 0);

    if (lines.length === 0) {
      els.batchSummary.hidden = true;
      els.batchTbody.replaceChildren();
      const tr = document.createElement("tr");
      tr.className = "placeholder-row";
      const td = document.createElement("td");
      td.colSpan = 6;
      td.textContent = "Chưa có dòng dữ liệu nào để phân tích.";
      tr.append(td);
      els.batchTbody.append(tr);
      return;
    }

    const mode = getSelectedMode();
    const tAll = performance.now();
    const rows = [];
    let posCount = 0;
    let tier1Count = 0;

    for (const line of lines) {
      const r = await classify(line, mode);
      if (r.is_address) posCount++;
      if (r.tier.startsWith("Tier 1 (Heuristic)")) tier1Count++;
      rows.push({ line, ...r });
    }

    const totalMs = performance.now() - tAll;
    const avgMs = totalMs / rows.length;

    // Summary strip
    els.batchSummary.hidden = false;
    els.batchSummary.replaceChildren();
    const items = [
      `Tổng: ${rows.length} dòng`,
      `Địa chỉ (TRUE): ${posCount}`,
      `Không phải địa chỉ (FALSE): ${rows.length - posCount}`,
      mode === "hybrid" ? `Tier 1 lọc nhanh: ${((tier1Count / rows.length) * 100).toFixed(0)}%` : `Chế độ: ${mode.toUpperCase()}`,
      `Tổng thời gian: ${totalMs.toFixed(2)} ms (TB ${avgMs.toFixed(2)} ms/dòng)`,
    ];
    for (const txt of items) {
      const span = document.createElement("span");
      span.textContent = txt;
      els.batchSummary.append(span);
    }

    // Populate table safely via DOM APIs (no innerHTML XSS risk)
    const frag = document.createDocumentFragment();
    for (const row of rows) {
      const tr = document.createElement("tr");

      const th = document.createElement("th");
      th.scope = "row";
      th.textContent = row.line;

      const tdDec = document.createElement("td");
      const badge = document.createElement("span");
      badge.className = `cell-badge ${row.is_address ? "verdict-pos" : "verdict-neg"}`;
      badge.textContent = row.is_address ? "✓ ĐỊA CHỈ" : "✕ KHÔNG PHẢI";
      tdDec.append(badge);

      const tdProb = document.createElement("td");
      tdProb.className = "cell-num";
      tdProb.textContent = `${(row.probability * 100).toFixed(2)}%`;

      const tdTier = document.createElement("td");
      tdTier.className = "cell-tier";
      tdTier.textContent = row.tier;

      const tdLat = document.createElement("td");
      tdLat.className = "cell-num";
      tdLat.textContent = `${row.latency_ms.toFixed(2)} ms`;

      const tdNote = document.createElement("td");
      tdNote.className = "cell-note";
      tdNote.textContent = row.reason;

      tr.append(th, tdDec, tdProb, tdTier, tdLat, tdNote);
      frag.append(tr);
    }

    els.batchTbody.replaceChildren(frag);
  }

  // Wire events
  els.singleForm.addEventListener("submit", (e) => {
    e.preventDefault();
    handleSingleRun();
  });

  // IME-safe Enter-to-submit for Vietnamese Telex/VNI keyboards
  els.singleInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      handleSingleRun();
    }
  });

  els.singleClear.addEventListener("click", () => {
    els.singleInput.value = "";
    els.singleResult.className = "result-box result-empty";
    els.singleResult.replaceChildren();
    const p = document.createElement("p");
    p.className = "empty-msg";
    p.textContent = "Đã xóa. Nhập chuỗi mới hoặc chọn mẫu nhanh để kiểm thử.";
    els.singleResult.append(p);
    els.singleInput.focus();
  });

  els.presetChips.addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-sample]");
    if (!btn) return;
    els.singleInput.value = btn.dataset.sample || "";
    handleSingleRun();
  });

  els.batchForm.addEventListener("submit", (e) => {
    e.preventDefault();
    handleBatchRun();
  });

  els.batchLoadSample.addEventListener("click", () => {
    els.batchInput.value = DEFAULT_BATCH_SAMPLES.join("\n");
  });

  els.batchClear.addEventListener("click", () => {
    els.batchInput.value = "";
    els.batchSummary.hidden = true;
    els.batchTbody.replaceChildren();
    const tr = document.createElement("tr");
    tr.className = "placeholder-row";
    const td = document.createElement("td");
    td.colSpan = 6;
    td.textContent = "Danh sách trống. Bấm 'Nạp 14 mẫu chuẩn' hoặc dán văn bản của bạn.";
    tr.append(td);
    els.batchTbody.append(tr);
  });

  // Pre-populate batch input with curated sample set
  els.batchInput.value = DEFAULT_BATCH_SAMPLES.join("\n");

  // Boot ONNX WASM engine
  initEngine();
})();
