/**
 * Laya Address VN v3 — Zero-Dependency Embeddable Address Validator,
 * Telex/Fuzzy Recovery, 2025 2-Tier Converter, and <laya-address-input> Widget.
 *
 * UMD module: exports `window.LayaAddressSDK` in browsers and `module.exports` in Node.js.
 * Zero CDN, zero external CSS required.
 */
(function (root) {
  "use strict";

  const isBrowser = typeof window !== "undefined" && typeof document !== "undefined";
  const isNode = typeof module !== "undefined" && !!module.exports && typeof require !== "undefined";

  let Heuristics = isBrowser ? root.LayaHeuristics || null : null;
  let TelexFuzzy = isBrowser ? root.LayaTelexFuzzy || null : null;

  const state = {
    initialized: false,
    initPromise: null,
    basePath: "",
    gazetteerReady: false,
    converterReady: false,
    onnxReady: false,
    useOnnx: true,
    session: null,
    vocab: null,
    calibration: null,
    loadTimeMs: 0,
    onnxError: null,
  };

  function resolveDeps() {
    if (isNode) {
      const path = require("path");
      if (!Heuristics) {
        Heuristics = require(path.join(__dirname, "..", "dist", "heuristics.js"));
      }
      if (!TelexFuzzy) {
        TelexFuzzy = require(path.join(__dirname, "laya-telex-fuzzy.js"));
      }
    } else {
      if (!Heuristics && root.LayaHeuristics) Heuristics = root.LayaHeuristics;
      if (!TelexFuzzy && root.LayaTelexFuzzy) TelexFuzzy = root.LayaTelexFuzzy;
    }
  }

  function detectBrowserBasePath() {
    if (!isBrowser) return "";
    if (document.currentScript && document.currentScript.src) {
      try {
        const u = new URL(document.currentScript.src, window.location.href);
        return u.href.replace(/\/sdk\/[^/]+$/, "/dist");
      } catch (_) {
        // Ignore and fallback
      }
    }
    const scripts = document.querySelectorAll('script[src*="laya-address-widget"]');
    if (scripts.length > 0) {
      try {
        const u = new URL(scripts[scripts.length - 1].src, window.location.href);
        return u.href.replace(/\/sdk\/[^/]+$/, "/dist");
      } catch (_) {
        // Ignore
      }
    }
    return "../dist";
  }

  const DEFAULT_BROWSER_BASE = detectBrowserBasePath();

  function ensureNodeInitSync(options) {
    resolveDeps();
    if (state.initialized) return true;
    if (!isNode) return false;

    const fs = require("fs");
    const path = require("path");
    const opts = options || {};
    const distDir = opts.basePath || path.join(__dirname, "..", "dist");
    const modelDir = path.join(distDir, "model");

    if (opts.heuristics) Heuristics = opts.heuristics;
    if (opts.telexFuzzy) TelexFuzzy = opts.telexFuzzy;

    const gazJson = opts.gazetteer || JSON.parse(fs.readFileSync(path.join(modelDir, "vn_admin_gazetteer.json"), "utf8"));
    const convJson = opts.converter || JSON.parse(fs.readFileSync(path.join(modelDir, "vn_admin_v3_converter.json"), "utf8"));

    const vocabPath = path.join(modelDir, "vocab.json");
    const calibPath = path.join(modelDir, "calibration.json");
    if (!state.vocab && (opts.vocab || fs.existsSync(vocabPath))) {
      state.vocab = opts.vocab || JSON.parse(fs.readFileSync(vocabPath, "utf8"));
    }
    if (!state.calibration && (opts.calibration || fs.existsSync(calibPath))) {
      state.calibration = opts.calibration || JSON.parse(fs.readFileSync(calibPath, "utf8"));
    }

    Heuristics.initGazetteer(gazJson);
    TelexFuzzy.initConverterV3(convJson);
    state.basePath = distDir;
    state.gazetteerReady = true;
    state.converterReady = true;
    state.useOnnx = opts.useOnnx === true;
    state.initialized = true;
    return true;
  }

  /**
   * Initializes Tier-1 Gazetteer, Telex/Fuzzy v3 Converter, and optionally local ONNX WASM.
   */
  async function init(options) {
    const opts = options || {};
    if (state.initialized && !opts.force) {
      return getStats();
    }
    if (state.initPromise && !opts.force) {
      return state.initPromise;
    }

    state.initPromise = (async () => {
      const t0 = typeof performance !== "undefined" ? performance.now() : Date.now();
      resolveDeps();
      if (opts.heuristics) Heuristics = opts.heuristics;
      if (opts.telexFuzzy) TelexFuzzy = opts.telexFuzzy;

      if (!Heuristics || !TelexFuzzy) {
        throw new Error("Missing LayaHeuristics or LayaTelexFuzzy dependency");
      }

      if (isNode && !opts.fetchMode) {
        ensureNodeInitSync(opts);
        state.loadTimeMs = +(((typeof performance !== "undefined" ? performance.now() : Date.now()) - t0)).toFixed(2);
        return getStats();
      }

      const base = (opts.basePath || DEFAULT_BROWSER_BASE || "../dist").replace(/\/+$/, "");
      state.basePath = base;
      state.useOnnx = opts.useOnnx !== false;

      const fetchJson = async (url) => {
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`HTTP ${resp.status} loading ${url}`);
        return resp.json();
      };

      const [gazJson, convJson, vocabJson, calibJson] = await Promise.all([
        opts.gazetteer ? Promise.resolve(opts.gazetteer) : fetchJson(`${base}/model/vn_admin_gazetteer.json`),
        opts.converter ? Promise.resolve(opts.converter) : fetchJson(`${base}/model/vn_admin_v3_converter.json`),
        opts.vocab ? Promise.resolve(opts.vocab) : fetchJson(`${base}/model/vocab.json`).catch(() => null),
        opts.calibration ? Promise.resolve(opts.calibration) : fetchJson(`${base}/model/calibration.json`).catch(() => null),
      ]);

      Heuristics.initGazetteer(gazJson);
      TelexFuzzy.initConverterV3(convJson);
      state.vocab = vocabJson;
      state.calibration = calibJson;
      state.gazetteerReady = true;
      state.converterReady = true;

      // Optional Tier-2 ONNX WASM initialization when `ort` is available
      const ortLib = opts.ort || (isBrowser ? root.ort : null);
      if (state.useOnnx && ortLib && state.vocab && state.calibration) {
        try {
          const wasmPaths = opts.wasmPaths || `${base}/vendor/onnxruntime-web/`;
          ortLib.env.wasm.wasmPaths = wasmPaths;
          ortLib.env.wasm.numThreads = 1;
          ortLib.env.wasm.simd = true;

          const onnxUrl = opts.onnxPath || `${base}/model/laya_address_vn.onnx`;
          state.session = await ortLib.InferenceSession.create(onnxUrl, {
            executionProviders: ["wasm"],
          });
          // Warmup inference
          const warmData = new BigInt64Array(state.vocab.max_length);
          const warmTensor = new ortLib.Tensor("int64", warmData, [1, state.vocab.max_length]);
          await state.session.run({ input_ids: warmTensor });
          state.onnxReady = true;
          state.onnxError = null;
        } catch (err) {
          state.onnxReady = false;
          state.onnxError = err instanceof Error ? err.message : String(err);
        }
      }

      state.loadTimeMs = +(((typeof performance !== "undefined" ? performance.now() : Date.now()) - t0)).toFixed(2);
      state.initialized = true;
      return getStats();
    })();

    return state.initPromise;
  }

  function getStats() {
    return {
      initialized: state.initialized,
      gazetteerReady: state.gazetteerReady,
      converterReady: state.converterReady,
      onnxReady: state.onnxReady,
      onnxError: state.onnxError,
      loadTimeMs: state.loadTimeMs,
      calibration: state.calibration,
    };
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

  function tokenizeToInt64(text) {
    const { char_to_id, unk_id, max_length } = state.vocab;
    const norm = (text || "").normalize("NFC").trim().replace(/\s+/g, " ").toLowerCase();
    const ids = new BigInt64Array(max_length);
    const limit = Math.min(norm.length, max_length);
    for (let i = 0; i < limit; i++) {
      const ch = norm[i];
      const id = Object.prototype.hasOwnProperty.call(char_to_id, ch) ? char_to_id[ch] : unk_id;
      ids[i] = BigInt(id);
    }
    return ids;
  }

  async function runOnnxTier2(repairedText) {
    const ortLib = isBrowser ? root.ort : null;
    if (!state.onnxReady || !state.session || !ortLib) return null;
    const ids = tokenizeToInt64(repairedText);
    const tensor = new ortLib.Tensor("int64", ids, [1, state.vocab.max_length]);
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
    return {
      is_address: isAddress,
      sufficiency: classes[bestIdx],
      probability: gateProb,
      sufficiency_probs: sufProbs,
    };
  }

  function buildMissingHint(rawText, t1Reason, fuzzyMatch) {
    const norm = TelexFuzzy.normalizeText(rawText);
    if (fuzzyMatch && fuzzyMatch.matched) {
      if (fuzzyMatch.match_type === "district_province") {
        return `Thiếu Phường/Xã hoặc Số nhà/Tên đường (${fuzzyMatch.district}, ${fuzzyMatch.province})`;
      }
      if (fuzzyMatch.match_type === "single_ward") {
        return `Chỉ có ${fuzzyMatch.ward} — cần bổ sung Quận/Huyện hoặc Tỉnh/Thành phố`;
      }
    }
    if (/\b(?:ngo|ngach|hem|kiet|thon|ap|xom|to|duong|so)\b/.test(norm)) {
      return `Thiếu Phường/Xã và Tỉnh/Thành phố (ví dụ: ${rawText.trim()}, Phường Hồng Hà, Hà Nội)`;
    }
    return t1Reason || "Địa chỉ chưa đủ cấp Phường/Xã + Tỉnh/TP";
  }

  function statusMessageVi(sufficiency, conv, telexFixed, fuzzyFixed) {
    if (sufficiency === "address_sufficient") {
      if (conv && conv.converted && conv.new_formatted) {
        return fuzzyFixed
          ? `Địa chỉ hợp lệ (khôi phục lỗi gõ → chuẩn 2025: ${conv.new_admin_formatted})`
          : `Địa chỉ hợp lệ (chuẩn 2025: ${conv.new_admin_formatted})`;
      }
      if (telexFixed) return "Địa chỉ hợp lệ (đã tự động sửa lỗi gõ Telex)";
      return "Địa chỉ hợp lệ — Đủ cấp giao hàng";
    }
    if (sufficiency === "address_insufficient") {
      return "Địa chỉ thiếu thông tin — Chọn gợi ý bên dưới hoặc bổ sung thêm";
    }
    return "Không phải địa chỉ giao hàng hợp lệ";
  }

  function finalizeResult(t0, payload) {
    const now = typeof performance !== "undefined" ? performance.now() : Date.now();
    return {
      raw_text: payload.raw_text,
      repaired_text: payload.repaired_text,
      is_address: payload.is_address,
      sufficiency: payload.sufficiency,
      missing_field_hint: payload.missing_field_hint || null,
      tier: payload.tier,
      conversion_2025: payload.conversion_2025 || null,
      suggestions: payload.suggestions || [],
      message_vi: payload.message_vi,
      latency_ms: +(now - t0).toFixed(3),
    };
  }

  function isHardNonAddressReason(reason) {
    if (!reason) return false;
    return (
      reason.startsWith("Chuỗi rỗng") ||
      reason.startsWith("Quá ngắn") ||
      reason.startsWith("Định dạng") ||
      reason.startsWith("Số điện thoại") ||
      reason.startsWith("Mã vận đơn") ||
      reason.startsWith("Chỉ chứa số") ||
      reason.startsWith("Họ tên người Việt") ||
      reason.startsWith("Cụm từ đơn hàng")
    );
  }

  /**
   * Synchronous Tier-1 + Telex + 1-Edit Fuzzy + 2025 Converter evaluation.
   */
  function evaluateInternal(rawText, allowAsyncOnnx) {
    const t0 = typeof performance !== "undefined" ? performance.now() : Date.now();
    if (!state.initialized && isNode) {
      ensureNodeInitSync();
    }
    resolveDeps();

    const raw = rawText == null ? "" : String(rawText);
    const repaired = TelexFuzzy.repairTelexTypos(raw);
    const telexChanged = repaired !== raw;

    const t1 = Heuristics.evaluateTier1(repaired);

    // 1. Direct Tier-1 Sufficient
    if (t1.decision === "clear_sufficient") {
      const conv = TelexFuzzy.convertLegacyTo2025(repaired);
      const hasConv = conv && (conv.new_ward || (conv.candidates && conv.candidates.length > 0));
      return finalizeResult(t0, {
        raw_text: raw,
        repaired_text: repaired,
        is_address: true,
        sufficiency: "address_sufficient",
        missing_field_hint: null,
        tier: telexChanged ? "Tier 1 + Telex" : "Tier 1 (Gazetteer)",
        conversion_2025: hasConv ? conv : null,
        suggestions: [],
        message_vi: statusMessageVi("address_sufficient", hasConv ? conv : null, telexChanged, false),
      });
    }

    // 2. Hard Non-Address Fast Reject (empty, email, URL, phone, BBQ/chat trap, person name)
    if (t1.decision === "not_address" && isHardNonAddressReason(t1.reason)) {
      return finalizeResult(t0, {
        raw_text: raw,
        repaired_text: repaired,
        is_address: false,
        sufficiency: "not_address",
        missing_field_hint: null,
        tier: "Tier 1 (Rule Guard)",
        conversion_2025: null,
        suggestions: [],
        message_vi: `${statusMessageVi("not_address")} (${t1.reason})`,
      });
    }

    // 3. Soft Reject / Insufficient / Ambiguous -> Try 1-Edit Fuzzy Gazetteer Recovery
    const fuzzy = TelexFuzzy.fuzzyMatchGazetteer(repaired);
    const fuzzyResolvedPair =
      fuzzy.matched &&
      (fuzzy.match_type === "ward_district_province" ||
        fuzzy.match_type === "ward_district" ||
        fuzzy.match_type === "ward_province" ||
        fuzzy.match_type === "ward_province_2025" ||
        fuzzy.match_type === "island_special");

    if (fuzzyResolvedPair) {
      const t1Fuzzy = Heuristics.evaluateTier1(fuzzy.fuzzy_repaired_text);
      if (t1Fuzzy.decision === "clear_sufficient" || fuzzy.match_type === "island_special") {
        const tierLabel = fuzzy.edit_distance > 0
          ? "Tier 1 + Fuzzy (1-edit)"
          : telexChanged
            ? "Tier 1 + Telex"
            : "Tier 1 (ConverterV3)";
        return finalizeResult(t0, {
          raw_text: raw,
          repaired_text: repaired,
          is_address: true,
          sufficiency: "address_sufficient",
          missing_field_hint: null,
          tier: tierLabel,
          conversion_2025: fuzzy.conversion_2025,
          suggestions: [],
          message_vi: statusMessageVi("address_sufficient", fuzzy.conversion_2025, telexChanged, fuzzy.edit_distance > 0),
        });
      }
    }

    // 4. Insufficient Fragment (from Tier-1 OR recovered partial admin span like "loc tahnh")
    if (t1.decision === "clear_insufficient" || (fuzzy.matched && (fuzzy.match_type === "single_ward" || fuzzy.match_type === "district_province"))) {
      const suggestions = TelexFuzzy.suggestWards(repaired, 3);
      const hint = buildMissingHint(repaired, t1.reason, fuzzy);
      return finalizeResult(t0, {
        raw_text: raw,
        repaired_text: repaired,
        is_address: false,
        sufficiency: "address_insufficient",
        missing_field_hint: hint,
        tier: fuzzy.edit_distance > 0 ? "Tier 1 + Fuzzy (1-edit)" : "Tier 1 (Gazetteer)",
        conversion_2025: fuzzy.conversion_2025 || null,
        suggestions,
        message_vi: statusMessageVi("address_insufficient"),
      });
    }

    // 5. Soft Not-Address that had zero fuzzy admin match
    if (t1.decision === "not_address") {
      return finalizeResult(t0, {
        raw_text: raw,
        repaired_text: repaired,
        is_address: false,
        sufficiency: "not_address",
        missing_field_hint: null,
        tier: "Tier 1 (Gazetteer/Rule)",
        conversion_2025: null,
        suggestions: [],
        message_vi: `${statusMessageVi("not_address")} (${t1.reason})`,
      });
    }

    // 6. Ambiguous -> Tier 2 ONNX WASM (if loaded & async allowed), else Heuristic fallback
    if (allowAsyncOnnx && state.onnxReady && state.session) {
      return runOnnxTier2(repaired).then((t2) => {
        const suf = t2.sufficiency;
        const isAddr = suf === "address_sufficient" || (t2.is_address && suf !== "address_insufficient");
        const suggestions = suf === "address_insufficient" ? TelexFuzzy.suggestWards(repaired, 3) : [];
        const conv = isAddr ? TelexFuzzy.convertLegacyTo2025(repaired, fuzzy) : null;
        return finalizeResult(t0, {
          raw_text: raw,
          repaired_text: repaired,
          is_address: isAddr,
          sufficiency: suf,
          missing_field_hint: suf === "address_insufficient" ? buildMissingHint(repaired, null, fuzzy) : null,
          tier: "Tier 2 (ONNX WASM)",
          conversion_2025: conv && conv.new_ward ? conv : null,
          suggestions,
          message_vi: statusMessageVi(suf, conv, telexChanged, false),
        });
      });
    }

    const fb = Heuristics.evaluateHeuristicOnly(repaired);
    const fbSuf = fb.sufficiency;
    const suggestions = fbSuf === "address_insufficient" ? TelexFuzzy.suggestWards(repaired, 3) : [];
    return finalizeResult(t0, {
      raw_text: raw,
      repaired_text: repaired,
      is_address: fb.is_address,
      sufficiency: fbSuf,
      missing_field_hint: fbSuf === "address_insufficient" ? buildMissingHint(repaired, fb.reason, fuzzy) : null,
      tier: "Tier 1 Fallback",
      conversion_2025: null,
      suggestions,
      message_vi: statusMessageVi(fbSuf, null, telexChanged, false),
    });
  }

  function evaluate(rawText) {
    return evaluateInternal(rawText, true);
  }

  function evaluateSync(rawText) {
    return evaluateInternal(rawText, false);
  }

  async function evaluateAsync(rawText) {
    return evaluateInternal(rawText, true);
  }

  // ---------------------------------------------------------------------------
  // Shared Widget Styles (Self-Contained, Zero External CSS, Light + Dark Safe)
  // ---------------------------------------------------------------------------
  const WIDGET_CSS = `
.laya-widget-panel {
  margin-top: 6px;
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
  font-size: 13px;
  line-height: 1.45;
  display: flex;
  flex-direction: column;
  gap: 6px;
  color: inherit;
}
.laya-widget-panel[hidden] { display: none; }
.laya-status-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}
.laya-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 3px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 600;
  border: 1px solid transparent;
}
.laya-pill-ok {
  background: rgba(22, 163, 74, 0.14);
  color: #16a34a;
  border-color: rgba(22, 163, 74, 0.35);
}
.laya-pill-warn {
  background: rgba(217, 119, 6, 0.14);
  color: #d97706;
  border-color: rgba(217, 119, 6, 0.35);
}
.laya-pill-err {
  background: rgba(225, 29, 72, 0.14);
  color: #e11d48;
  border-color: rgba(225, 29, 72, 0.35);
}
@media (prefers-color-scheme: dark) {
  .laya-pill-ok { color: #4ade80; border-color: rgba(74, 222, 128, 0.35); }
  .laya-pill-warn { color: #fbbf24; border-color: rgba(251, 191, 36, 0.35); }
  .laya-pill-err { color: #fb7185; border-color: rgba(251, 113, 133, 0.35); }
}
.laya-meta-tag {
  font-size: 11.5px;
  opacity: 0.72;
}
.laya-hint {
  font-size: 12.5px;
  color: #b45309;
  background: rgba(245, 158, 11, 0.1);
  border-left: 3px solid #f59e0b;
  padding: 4px 8px;
  border-radius: 4px;
}
@media (prefers-color-scheme: dark) {
  .laya-hint { color: #fcd34d; }
}
.laya-conv-badge {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 6px 10px;
  border-radius: 8px;
  background: rgba(14, 165, 233, 0.1);
  border: 1px solid rgba(14, 165, 233, 0.3);
  font-size: 12.5px;
}
.laya-conv-text strong {
  color: #0284c7;
}
@media (prefers-color-scheme: dark) {
  .laya-conv-text strong { color: #38bdf8; }
}
.laya-apply-btn, .laya-suggestion-chip {
  appearance: none;
  font: inherit;
  font-size: 12px;
  cursor: pointer;
  border-radius: 6px;
  padding: 4px 9px;
  border: 1px solid rgba(14, 165, 233, 0.45);
  background: rgba(14, 165, 233, 0.12);
  color: inherit;
  transition: background 0.12s ease, border-color 0.12s ease;
  text-align: left;
}
.laya-apply-btn:hover, .laya-suggestion-chip:hover {
  background: rgba(14, 165, 233, 0.24);
  border-color: rgba(14, 165, 233, 0.75);
}
.laya-suggestions {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
`;

  function ensureDocumentStyles() {
    if (!isBrowser || !document.head) return;
    if (document.getElementById("laya-widget-v3-styles")) return;
    const style = document.createElement("style");
    style.id = "laya-widget-v3-styles";
    style.textContent = WIDGET_CSS;
    document.head.appendChild(style);
  }

  function setNativeInputValue(inputEl, value) {
    if (!inputEl) return;
    const proto = inputEl instanceof HTMLTextAreaElement
      ? HTMLTextAreaElement.prototype
      : HTMLInputElement.prototype;
    const desc = Object.getOwnPropertyDescriptor(proto, "value");
    if (desc && typeof desc.set === "function") {
      desc.set.call(inputEl, value);
    } else {
      inputEl.value = value;
    }
    inputEl.dispatchEvent(new Event("input", { bubbles: true, composed: true }));
    inputEl.dispatchEvent(new Event("change", { bubbles: true, composed: true }));
  }

  function renderPanelContent(panel, result, onSelectText) {
    panel.replaceChildren();
    if (!result || !result.raw_text.trim()) {
      panel.hidden = true;
      return;
    }
    panel.hidden = false;

    const statusRow = document.createElement("div");
    statusRow.className = "laya-status-row";

    const pill = document.createElement("span");
    const tone =
      result.sufficiency === "address_sufficient"
        ? "ok"
        : result.sufficiency === "address_insufficient"
          ? "warn"
          : "err";
    pill.className = `laya-pill laya-pill-${tone}`;
    const icon = tone === "ok" ? "✓" : tone === "warn" ? "⚠" : "✕";
    const label =
      tone === "ok"
        ? "Hợp lệ (PASS)"
        : tone === "warn"
          ? "Thiếu thông tin"
          : "Không phải địa chỉ";
    pill.textContent = `${icon} ${label}`;

    const meta = document.createElement("span");
    meta.className = "laya-meta-tag";
    meta.textContent = `${result.tier} • ${result.latency_ms.toFixed(2)} ms`;

    statusRow.append(pill, meta);

    if (result.repaired_text && result.repaired_text !== result.raw_text) {
      const repairTag = document.createElement("span");
      repairTag.className = "laya-meta-tag";
      repairTag.textContent = `Sửa gõ: "${result.repaired_text}"`;
      statusRow.append(repairTag);
    }

    panel.append(statusRow);

    if (result.missing_field_hint) {
      const hint = document.createElement("div");
      hint.className = "laya-hint";
      hint.textContent = result.missing_field_hint;
      panel.append(hint);
    }

    if (result.conversion_2025 && result.conversion_2025.new_ward && result.conversion_2025.new_province) {
      const conv = result.conversion_2025;
      const badge = document.createElement("div");
      badge.className = "laya-conv-badge";

      const txt = document.createElement("div");
      txt.className = "laya-conv-text";
      const prefix = conv.already_2025 ? "Chuẩn 2025 (2 cấp): " : "Chuẩn 2025: ";
      txt.textContent = prefix;
      const strong = document.createElement("strong");
      strong.textContent = conv.new_formatted || `${conv.new_ward}, ${conv.new_province}`;
      txt.append(strong);
      if (conv.old_formatted && conv.converted) {
        const oldSpan = document.createElement("span");
        oldSpan.className = "laya-meta-tag";
        oldSpan.textContent = ` (cũ: ${conv.old_formatted})`;
        txt.append(oldSpan);
      }
      badge.append(txt);

      if (conv.new_formatted && conv.new_formatted !== result.raw_text.trim() && onSelectText) {
        const applyBtn = document.createElement("button");
        applyBtn.type = "button";
        applyBtn.className = "laya-apply-btn";
        applyBtn.textContent = "Áp dụng chuẩn 2025";
        applyBtn.addEventListener("click", () => onSelectText(conv.new_formatted));
        badge.append(applyBtn);
      }
      panel.append(badge);
    }

    if (Array.isArray(result.suggestions) && result.suggestions.length > 0) {
      const list = document.createElement("div");
      list.className = "laya-suggestions";
      list.setAttribute("role", "group");
      list.setAttribute("aria-label", "Gợi ý Phường/Xã chuẩn 2025");

      for (const s of result.suggestions) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "laya-suggestion-chip";
        btn.textContent = `+ ${s.label}`;
        btn.addEventListener("click", () => {
          if (onSelectText) onSelectText(s.insert_text);
        });
        list.append(btn);
      }
      panel.append(list);
    }
  }

  let panelIdCounter = 0;

  /**
   * Attaches live validation, 2025 badge, and 1-click Top-3 suggestion chips to any <input>/<textarea>.
   */
  function attachToInput(inputElement, options) {
    if (!isBrowser) throw new Error("attachToInput requires a browser DOM environment");
    const target = typeof inputElement === "string"
      ? document.querySelector(inputElement)
      : inputElement;
    if (!target) throw new Error("Target input element not found");

    ensureDocumentStyles();
    const opts = options || {};
    const debounceMs = typeof opts.debounceMs === "number" ? opts.debounceMs : 60;

    const panel = document.createElement("div");
    panel.className = "laya-widget-panel";
    panel.id = `laya-panel-${++panelIdCounter}`;
    panel.setAttribute("role", "status");
    panel.setAttribute("aria-live", "polite");
    panel.hidden = true;

    if (opts.container) {
      opts.container.appendChild(panel);
    } else if (target.parentNode) {
      target.parentNode.insertBefore(panel, target.nextSibling);
    }

    const prevDesc = target.getAttribute("aria-describedby");
    if (!prevDesc || !prevDesc.split(/\s+/).includes(panel.id)) {
      target.setAttribute("aria-describedby", prevDesc ? `${prevDesc} ${panel.id}` : panel.id);
    }

    let timer = null;
    let lastResult = null;
    let destroyed = false;
    let evalSeq = 0;

    const applyAutofill = (val) => {
      setNativeInputValue(target, val);
      target.focus();
      runEval();
    };

    const runEval = async () => {
      if (destroyed) return;
      const seq = ++evalSeq;
      if (!state.initialized) {
        await init(opts.initOptions || {});
      }
      if (destroyed || seq !== evalSeq) return;

      const text = target.value || "";
      if (!text.trim()) {
        lastResult = null;
        panel.replaceChildren();
        panel.hidden = true;
        if (typeof opts.onResult === "function") opts.onResult(null);
        return null;
      }

      const res = await evaluate(text);
      if (destroyed || seq !== evalSeq) return res;
      lastResult = res;
      renderPanelContent(panel, res, applyAutofill);
      if (typeof opts.onResult === "function") opts.onResult(res);
      return res;
    };

    const onInput = () => {
      clearTimeout(timer);
      if (!target.value.trim()) {
        runEval();
        return;
      }
      timer = setTimeout(runEval, debounceMs);
    };

    const onChange = () => {
      clearTimeout(timer);
      runEval();
    };

    target.addEventListener("input", onInput);
    target.addEventListener("change", onChange);

    if (target.value && target.value.trim()) {
      runEval();
    }

    return {
      evaluate: runEval,
      getLastResult: () => lastResult,
      panel,
      destroy() {
        destroyed = true;
        clearTimeout(timer);
        target.removeEventListener("input", onInput);
        target.removeEventListener("change", onChange);
        panel.remove();
      },
    };
  }

  // ---------------------------------------------------------------------------
  // Form-Associated Custom Element: <laya-address-input>
  // ---------------------------------------------------------------------------
  if (isBrowser && typeof HTMLElement !== "undefined" && typeof customElements !== "undefined") {
    const hasFace =
      "attachInternals" in HTMLElement.prototype &&
      typeof ElementInternals !== "undefined" &&
      "setFormValue" in ElementInternals.prototype;

    const SHADOW_CSS = `
:host {
  display: block;
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
}
:host([hidden]) {
  display: none;
}
.laya-ce-wrap {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.laya-ce-label {
  font-size: 13px;
  font-weight: 600;
}
.laya-ce-input {
  width: 100%;
  box-sizing: border-box;
  font: inherit;
  font-size: 14px;
  padding: 9px 12px;
  border-radius: 8px;
  border: 1px solid var(--laya-border, #64748b);
  background: var(--laya-input-bg, transparent);
  color: inherit;
}
.laya-ce-input:focus {
  outline: 2px solid var(--laya-accent, #0ea5e9);
  outline-offset: 1px;
}
.laya-ce-input:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
${WIDGET_CSS}
`;

    class LayaAddressInput extends HTMLElement {
      static formAssociated = true;
      static observedAttributes = [
        "value",
        "placeholder",
        "name",
        "required",
        "disabled",
        "readonly",
        "label",
        "autocomplete",
      ];

      #internals = null;
      #value = "";
      #timer = null;
      #evalSeq = 0;
      #lastResult = null;
      #inputEl = null;
      #labelEl = null;
      #panelEl = null;
      #rendered = false;

      constructor() {
        super();
        if (hasFace) {
          this.#internals = this.attachInternals();
        }
        this.attachShadow({ mode: "open", delegatesFocus: true });
      }

      connectedCallback() {
        this.#ensureRendered();
        if (this.hasAttribute("value") && !this.#value) {
          this.#value = this.getAttribute("value") || "";
        }
        this.#syncAllAttributes();
        this.#syncFormValue();
        if (!state.initialized) {
          const basePath = this.getAttribute("base-path") || undefined;
          init({ basePath }).then(() => {
            if (this.#value.trim()) this.evaluate();
          }).catch(() => {});
        } else if (this.#value.trim()) {
          this.evaluate();
        }
      }

      disconnectedCallback() {
        clearTimeout(this.#timer);
      }

      attributeChangedCallback(name, oldVal, newVal) {
        if (oldVal === newVal) return;
        if (!this.#rendered) return;
        if (name === "value") {
          this.value = newVal || "";
        } else {
          this.#syncAllAttributes();
          this.#validate();
        }
      }

      get value() {
        return this.#value;
      }

      set value(v) {
        const next = v == null ? "" : String(v);
        this.#value = next;
        if (this.#inputEl && this.#inputEl.value !== next) {
          this.#inputEl.value = next;
        }
        this.#syncFormValue();
        this.#validate();
      }

      get lastResult() {
        return this.#lastResult;
      }

      formResetCallback() {
        this.value = this.getAttribute("value") || "";
        this.#lastResult = null;
        if (this.#panelEl) {
          this.#panelEl.replaceChildren();
          this.#panelEl.hidden = true;
        }
      }

      formDisabledCallback(disabled) {
        if (this.#inputEl) {
          this.#inputEl.disabled = !!disabled;
        }
      }

      formStateRestoreCallback(restored) {
        if (typeof restored === "string") {
          this.value = restored;
        }
      }

      #ensureRendered() {
        if (this.#rendered) return;
        const style = document.createElement("style");
        style.textContent = SHADOW_CSS;

        const wrap = document.createElement("div");
        wrap.className = "laya-ce-wrap";

        const label = document.createElement("label");
        label.className = "laya-ce-label";
        label.htmlFor = "laya-ce-field";
        label.hidden = true;

        const input = document.createElement("input");
        input.id = "laya-ce-field";
        input.className = "laya-ce-input";
        input.type = "text";
        input.setAttribute("part", "input");
        input.setAttribute("aria-describedby", "laya-ce-panel");

        const panel = document.createElement("div");
        panel.id = "laya-ce-panel";
        panel.className = "laya-widget-panel";
        panel.setAttribute("role", "status");
        panel.setAttribute("aria-live", "polite");
        panel.hidden = true;

        input.addEventListener("input", () => {
          this.#value = input.value;
          this.#syncFormValue();
          this.#validate();
          this.dispatchEvent(new InputEvent("input", { bubbles: true, composed: true, data: input.value }));
          clearTimeout(this.#timer);
          if (!this.#value.trim()) {
            this.evaluate();
          } else {
            this.#timer = setTimeout(() => this.evaluate(), 60);
          }
        });

        input.addEventListener("change", () => {
          clearTimeout(this.#timer);
          this.dispatchEvent(new Event("change", { bubbles: true, composed: true }));
          this.evaluate();
        });

        wrap.append(label, input, panel);
        this.shadowRoot.append(style, wrap);

        this.#labelEl = label;
        this.#inputEl = input;
        this.#panelEl = panel;
        this.#rendered = true;
      }

      #syncAllAttributes() {
        if (!this.#inputEl) return;
        const lbl = this.getAttribute("label");
        if (lbl) {
          this.#labelEl.textContent = lbl;
          this.#labelEl.hidden = false;
        } else {
          this.#labelEl.hidden = true;
        }
        this.#inputEl.placeholder = this.getAttribute("placeholder") || "Nhập địa chỉ nhận hàng (VD: Số 5, Lộc Thạnh...)";
        this.#inputEl.autocomplete = this.getAttribute("autocomplete") || "street-address";
        this.#inputEl.disabled = this.hasAttribute("disabled");
        this.#inputEl.readOnly = this.hasAttribute("readonly");
        this.#inputEl.required = this.hasAttribute("required");
        if (this.#inputEl.value !== this.#value) {
          this.#inputEl.value = this.#value;
        }
      }

      #syncFormValue() {
        if (this.#internals && typeof this.#internals.setFormValue === "function") {
          this.#internals.setFormValue(this.#value);
        }
      }

      #validate() {
        if (!this.#internals || typeof this.#internals.setValidity !== "function") return;
        if (this.hasAttribute("required") && !this.#value.trim()) {
          this.#internals.setValidity({ valueMissing: true }, "Vui lòng nhập địa chỉ giao hàng.", this.#inputEl);
        } else {
          this.#internals.setValidity({});
        }
      }

      async evaluate() {
        const seq = ++this.#evalSeq;
        if (!state.initialized) {
          const basePath = this.getAttribute("base-path") || undefined;
          await init({ basePath });
        }
        if (seq !== this.#evalSeq) return this.#lastResult;

        if (!this.#value.trim()) {
          this.#lastResult = null;
          this.#panelEl.replaceChildren();
          this.#panelEl.hidden = true;
          this.#validate();
          return null;
        }

        const res = await evaluate(this.#value);
        if (seq !== this.#evalSeq) return res;
        this.#lastResult = res;

        renderPanelContent(this.#panelEl, res, (insertText) => {
          this.value = insertText;
          this.dispatchEvent(new InputEvent("input", { bubbles: true, composed: true, data: insertText }));
          this.dispatchEvent(new Event("change", { bubbles: true, composed: true }));
          this.#inputEl.focus();
          this.evaluate();
        });

        this.dispatchEvent(new CustomEvent("laya-evaluate", { detail: res, bubbles: true, composed: true }));
        return res;
      }
    }

    if (!customElements.get("laya-address-input")) {
      customElements.define("laya-address-input", LayaAddressInput);
    }
  }

  const api = {
    init,
    getStats,
    evaluate,
    evaluateSync,
    evaluateAsync,
    attachToInput,
    repairTelexTypos: (t) => {
      resolveDeps();
      return TelexFuzzy ? TelexFuzzy.repairTelexTypos(t) : String(t || "");
    },
    convertLegacyTo2025: (t) => {
      if (!state.initialized && isNode) ensureNodeInitSync();
      resolveDeps();
      return TelexFuzzy ? TelexFuzzy.convertLegacyTo2025(t) : null;
    },
    suggestWards: (t, limit) => {
      if (!state.initialized && isNode) ensureNodeInitSync();
      resolveDeps();
      return TelexFuzzy ? TelexFuzzy.suggestWards(t, limit) : [];
    },
  };

  if (isNode) {
    module.exports = api;
  } else {
    root.LayaAddressSDK = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
