/**
 * Way A v2 Browser Controller:
 *  - Vendored local onnxruntime-web WASM runtime (zero CDN!)
 *  - Local GSO/2025 vn_admin_gazetteer.json (Tier 1)
 *  - Dual-output ONNX Char-CNN (logit + sufficiency_logits, Tier 2)
 */
(function () {
  "use strict";

  const DEFAULT_BATCH_SAMPLES = [
    "Lộc Thạnh, Lộc Ninh",
    "Lộc Thạnh Lộc Ninh",
    "loc thanh loc ninh",
    "x. lộc thạnh, h. lộc ninh",
    "p.12 q.tb",
    "Phường Hồng Hà, Hà Nội",
    "Số 123 Đường Nguyễn Trãi, Phường Thượng Đình, Quận Thanh Xuân, Hà Nội",
    "Xã Lộc Thạnh",
    "Thanh Xuân, Hà Nội",
    "ngõ 12",
    "Thôn 7",
    "73",
    "Cho mình 3 chai nướng BBQ",
    "bánh mì Sài Gòn",
    "Nguyễn Thị Thanh Xuân",
    "Phí ship về Bình Phước bao nhiêu?",
  ];

  const SUFFICIENCY_LABELS_VI = {
    address_sufficient: "Đủ thông tin giao hàng",
    address_insufficient: "Địa chỉ thiếu cấp",
    not_address: "Không phải địa chỉ",
  };

  const state = {
    session: null,
    vocab: null,
    calibration: null,
    gazetteerReady: false,
    onnxReady: false,
    loadError: null,
    loadTimeMs: null,
  };

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
    const c = Math.max(-30, Math.min(30, x));
    return 1 / (1 + Math.exp(-c));
  }

  function softmax(arr) {
    const maxVal = Math.max(...arr);
    const exps = arr.map((v) => Math.exp(Math.max(-50, Math.min(50, v - maxVal))));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((e) => e / sum);
  }

  async function initEngine() {
    const t0 = performance.now();
    try {
      const [vocabResp, calibResp, gazResp] = await Promise.all([
        fetch("./model/vocab.json"),
        fetch("./model/calibration.json"),
        fetch("./model/vn_admin_gazetteer.json"),
      ]);
      if (!vocabResp.ok || !calibResp.ok || !gazResp.ok) {
        throw new Error("Thiếu tệp trong ./model/ (vocab.json, calibration.json, hoặc vn_admin_gazetteer.json)");
      }
      state.vocab = await vocabResp.json();
      state.calibration = await calibResp.json();
      const gazJson = await gazResp.json();
      window.LayaHeuristics.initGazetteer(gazJson);
      state.gazetteerReady = true;

      const onnxKb = (state.calibration.model_size_bytes / 1024).toFixed(1);
      const totalMb = (state.calibration.total_artifacts_bytes / (1024 * 1024)).toFixed(2);
      els.metaSize.textContent = `${onnxKb} KB / ${totalMb} MB`;
      els.metaTemp.textContent = `T=${Number(state.calibration.temperature).toFixed(3)} / ${Number(state.calibration.sufficiency_temperature).toFixed(3)}`;

      const gateM = state.calibration.metrics?.test_gate_post;
      const sufM = state.calibration.metrics?.test_sufficiency_post;
      if (gateM && sufM) {
        els.metaAcc.textContent = `${(gateM.accuracy * 100).toFixed(2)}% / ${(sufM.accuracy * 100).toFixed(2)}%`;
        els.metaEce.textContent = `${(gateM.ece * 100).toFixed(2)}% / ${(sufM.ece * 100).toFixed(2)}%`;
      }

      if (typeof window.ort === "undefined") {
        throw new Error("Không tải được ./vendor/onnxruntime-web/ort.wasm.min.js");
      }

      window.ort.env.wasm.wasmPaths = "./vendor/onnxruntime-web/";
      window.ort.env.wasm.numThreads = 1;
      window.ort.env.wasm.simd = true;

      state.session = await window.ort.InferenceSession.create("./model/laya_address_vn.onnx", {
        executionProviders: ["wasm"],
      });

      const warmData = new BigInt64Array(state.vocab.max_length);
      const warmTensor = new window.ort.Tensor("int64", warmData, [1, state.vocab.max_length]);
      await state.session.run({ input_ids: warmTensor });

      state.loadTimeMs = +(performance.now() - t0).toFixed(0);
      state.onnxReady = true;
      els.metaLoad.textContent = `${state.loadTimeMs} ms`;
      setEngineStatus("ready", `Sẵn sàng • Local WASM + Gazetteer (${totalMb} MB)`);
    } catch (err) {
      state.onnxReady = false;
      state.loadError = err instanceof Error ? err.message : String(err);
      els.metaLoad.textContent = "Lỗi WASM";
      setEngineStatus("error", "WASM chưa sẵn sàng (chỉ Gazetteer)");
      showBanner(`Lưu ý ONNX WASM: ${state.loadError}`);
    }
  }

  async function runOnnxSingle(text) {
    if (!state.onnxReady || !state.session) {
      throw new Error(state.loadError || "Mô hình ONNX WASM chưa sẵn sàng.");
    }
    const t0 = performance.now();
    const ids = tokenizeToInt64(text);
    const tensor = new window.ort.Tensor("int64", ids, [1, state.vocab.max_length]);
    const outputs = await state.session.run({ input_ids: tensor });

    const rawLogit = Number(outputs.logit.data[0]);
    const temp = Number(state.calibration.temperature || 1.0);
    const threshold = Number(state.calibration.threshold ?? 0.5);
    const gateProb = sigmoid(rawLogit / temp);
    const isAddress = gateProb >= threshold;

    const sufRaw = Array.from(outputs.sufficiency_logits.data, Number);
    const tSuf = Number(state.calibration.sufficiency_temperature || 1.0);
    const sufProbs = softmax(sufRaw.map((v) => v / tSuf));
    const classes = state.calibration.sufficiency_classes || [
      "not_address",
      "address_insufficient",
      "address_sufficient",
    ];
    let bestIdx = 0;
    for (let i = 1; i < sufProbs.length; i++) {
      if (sufProbs[i] > sufProbs[bestIdx]) bestIdx = i;
    }
    const sufficiency = classes[bestIdx];
    const latencyMs = +(performance.now() - t0).toFixed(3);

    return {
      is_address: isAddress,
      gate_label: isAddress ? "PASS" : "BLOCK",
      sufficiency,
      sufficiency_vi: SUFFICIENCY_LABELS_VI[sufficiency] || sufficiency,
      probability: gateProb,
      sufficiency_probs: sufProbs,
      raw_logit: +rawLogit.toFixed(4),
      tier: "Tier 2 (Dual-Head ONNX WASM)",
      latency_ms: latencyMs,
      reason: `Char-CNN v2 • logit=${rawLogit.toFixed(2)}, T=${temp.toFixed(3)}, T_suf=${tSuf.toFixed(3)}`,
    };
  }

  async function classify(text, mode) {
    const H = window.LayaHeuristics;
    if (mode === "heuristic") {
      const r = H.evaluateHeuristicOnly(text);
      return {
        is_address: r.is_address,
        gate_label: r.is_address ? "PASS" : "BLOCK",
        sufficiency: r.sufficiency,
        sufficiency_vi: SUFFICIENCY_LABELS_VI[r.sufficiency] || r.sufficiency,
        probability: r.probability,
        sufficiency_probs: r.sufficiency_probs,
        tier: r.tier,
        latency_ms: r.latency_ms,
        reason: r.reason,
      };
    }

    if (mode === "onnx") {
      if (!state.onnxReady) {
        const fb = H.evaluateHeuristicOnly(text);
        return {
          is_address: fb.is_address,
          gate_label: fb.is_address ? "PASS" : "BLOCK",
          sufficiency: fb.sufficiency,
          sufficiency_vi: SUFFICIENCY_LABELS_VI[fb.sufficiency] || fb.sufficiency,
          probability: fb.probability,
          sufficiency_probs: fb.sufficiency_probs,
          tier: "Fallback Gazetteer (ONNX chưa sẵn sàng)",
          latency_ms: fb.latency_ms,
          reason: fb.reason,
        };
      }
      return await runOnnxSingle(text);
    }

    // Hybrid 2-Tier
    const t1 = H.evaluateTier1(text);
    if (t1.decision !== "ambiguous") {
      return {
        is_address: t1.is_address,
        gate_label: t1.is_address ? "PASS" : "BLOCK",
        sufficiency: t1.sufficiency,
        sufficiency_vi: SUFFICIENCY_LABELS_VI[t1.sufficiency] || t1.sufficiency,
        probability: t1.probability,
        sufficiency_probs: t1.sufficiency_probs,
        tier: t1.tier,
        latency_ms: t1.latency_ms,
        reason: t1.reason,
      };
    }

    if (!state.onnxReady) {
      const fb = H.evaluateHeuristicOnly(text);
      return {
        is_address: fb.is_address,
        gate_label: fb.is_address ? "PASS" : "BLOCK",
        sufficiency: fb.sufficiency,
        sufficiency_vi: SUFFICIENCY_LABELS_VI[fb.sufficiency] || fb.sufficiency,
        probability: fb.probability,
        sufficiency_probs: fb.sufficiency_probs,
        tier: "Tier 1 Fallback (ONNX chưa sẵn sàng)",
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

  function styleClassForSufficiency(sufficiency, isAddress) {
    if (isAddress || sufficiency === "address_sufficient") return "pos";
    if (sufficiency === "address_insufficient") return "warn";
    return "neg";
  }

  function renderSingleResult(result) {
    const box = els.singleResult;
    const tone = styleClassForSufficiency(result.sufficiency, result.is_address);
    box.classList.remove("result-empty", "result-pos", "result-warn", "result-neg");
    box.classList.add(`result-${tone}`);
    box.replaceChildren();

    const pct = (result.probability * 100).toFixed(2);

    const top = document.createElement("div");
    top.className = "result-top";

    const badges = document.createElement("div");
    badges.className = "badge-group";

    const gateBadge = document.createElement("span");
    gateBadge.className = `verdict-badge ${result.is_address ? "verdict-pos" : "verdict-neg"}`;
    gateBadge.textContent = `${result.is_address ? "✓" : "✕"} GATE ${result.gate_label}`;

    const sufBadge = document.createElement("span");
    sufBadge.className = `verdict-badge verdict-${tone}`;
    sufBadge.textContent = `${result.sufficiency} (${result.sufficiency_vi})`;

    badges.append(gateBadge, sufBadge);

    const probEl = document.createElement("span");
    probEl.className = "prob-display";
    probEl.textContent = `${pct}%`;

    top.append(badges, probEl);

    const track = document.createElement("div");
    track.className = "prob-track";
    track.setAttribute("aria-hidden", "true");
    const fill = document.createElement("div");
    fill.className = `prob-fill prob-fill-${tone}`;
    fill.style.inlineSize = `${Math.max(2, Math.min(100, result.probability * 100))}%`;
    track.append(fill);

    const sufRow = document.createElement("div");
    sufRow.className = "suf-breakdown";
    if (Array.isArray(result.sufficiency_probs) && result.sufficiency_probs.length === 3) {
      const names = ["not_address", "address_insufficient", "address_sufficient"];
      for (let i = 0; i < 3; i++) {
        const sp = document.createElement("span");
        sp.textContent = `${names[i]}: ${(result.sufficiency_probs[i] * 100).toFixed(2)}%`;
        sufRow.append(sp);
      }
    }

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

    box.append(top, track, sufRow, meta);
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
      td.colSpan = 7;
      td.textContent = "Chưa có dòng dữ liệu nào để phân tích.";
      tr.append(td);
      els.batchTbody.append(tr);
      return;
    }

    const mode = getSelectedMode();
    const tAll = performance.now();
    const rows = [];
    let passCount = 0;
    let insuffCount = 0;
    let tier1Count = 0;

    for (const line of lines) {
      const r = await classify(line, mode);
      if (r.is_address) passCount++;
      if (r.sufficiency === "address_insufficient") insuffCount++;
      if (r.tier.startsWith("Tier 1 (")) tier1Count++;
      rows.push({ line, ...r });
    }

    const totalMs = performance.now() - tAll;
    const avgMs = totalMs / rows.length;

    els.batchSummary.hidden = false;
    els.batchSummary.replaceChildren();
    const items = [
      `Tổng: ${rows.length} dòng`,
      `PASS (sufficient): ${passCount}`,
      `BLOCK insufficient: ${insuffCount}`,
      `BLOCK not_address: ${rows.length - passCount - insuffCount}`,
      mode === "hybrid" ? `Tier 1 xử lý: ${((tier1Count / rows.length) * 100).toFixed(0)}%` : `Chế độ: ${mode.toUpperCase()}`,
      `Tổng: ${totalMs.toFixed(2)} ms (TB ${avgMs.toFixed(2)} ms/dòng)`,
    ];
    for (const txt of items) {
      const span = document.createElement("span");
      span.textContent = txt;
      els.batchSummary.append(span);
    }

    const frag = document.createDocumentFragment();
    for (const row of rows) {
      const tr = document.createElement("tr");
      const tone = styleClassForSufficiency(row.sufficiency, row.is_address);

      const th = document.createElement("th");
      th.scope = "row";
      th.textContent = row.line;

      const tdGate = document.createElement("td");
      const gateBadge = document.createElement("span");
      gateBadge.className = `cell-badge ${row.is_address ? "verdict-pos" : "verdict-neg"}`;
      gateBadge.textContent = row.is_address ? "✓ PASS" : "✕ BLOCK";
      tdGate.append(gateBadge);

      const tdSuf = document.createElement("td");
      const sufBadge = document.createElement("span");
      sufBadge.className = `cell-badge verdict-${tone}`;
      sufBadge.textContent = row.sufficiency;
      tdSuf.append(sufBadge);

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

      tr.append(th, tdGate, tdSuf, tdProb, tdTier, tdLat, tdNote);
      frag.append(tr);
    }

    els.batchTbody.replaceChildren(frag);
  }

  els.singleForm.addEventListener("submit", (e) => {
    e.preventDefault();
    handleSingleRun();
  });

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
    td.colSpan = 7;
    td.textContent = "Danh sách trống. Bấm 'Nạp bộ mẫu v2' hoặc dán văn bản của bạn.";
    tr.append(td);
    els.batchTbody.append(tr);
  });

  els.batchInput.value = DEFAULT_BATCH_SAMPLES.join("\n");
  initEngine();
})();
