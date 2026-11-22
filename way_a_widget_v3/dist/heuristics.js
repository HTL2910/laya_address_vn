/**
 * Tier 1: Fast Vietnamese Address Gazetteer & Structural Pre-Filter (< 1 ms)
 *
 * Uses `vn_admin_gazetteer.json` (63 provinces, 656 districts, 7,044 wards,
 * legacy 3-tier + 2025 2-tier edges) to classify:
 *   - `clear_sufficient`  (sufficiency = "address_sufficient",   is_address = true)
 *   - `clear_insufficient`(sufficiency = "address_insufficient", is_address = false)
 *   - `not_address`       (sufficiency = "not_address",          is_address = false)
 *   - `ambiguous`         (route to Tier-2 dual-output ONNX Char-CNN)
 */
(function (root) {
  "use strict";

  const SURNAMES_RE = /^(?:nguyen|tran|le|pham|hoang|huynh|phan|vu|vo|dang|bui|do|ho|ngo|duong|ly)\b/;
  const MIDDLE_NAMES_RE = /^(?:van|thi|minh|quoc|thanh|hoang|ngoc|thu|anh|hai|duc)$/;

  const HISTORICAL_PERSON_TRAPS = new Set([
    "nguyen trai", "le loi", "tran hung dao", "hai ba trung", "ly thuong kiet",
    "nguyen hue", "quang trung", "dinh tien hoang", "phan boi chau", "phan chu trinh",
    "nguyen thi minh khai", "ton duc thang", "vo nguyen giap", "le duan", "hoang dieu",
    "ngo quyen", "ly tu trong", "nguyen du", "ho xuan huong", "nguyen cong tru",
    "nguyen van cu", "tran phu", "vo thi sau", "nguyen dinh chieu", "le quy don",
    "hung vuong", "pham ngu lao", "tran quoc toan", "le hong phong", "hoang hoa tham",
  ]);

  const PRODUCT_UNITS_RE = /(?<![/.\d])\b\d+\s*(?:g|gr|kg|ml|lit|cai|chiec|goi|hop|bich|tui|chai|hu|thung|suat|vien|cuon|quyen|cm|mm|gb|tb|mah)\b/i;

  const SPECIALTY_FOOD_PRODUCT_RE = /\b(?:banh mi|banh dau xanh|nhan long|com tam|banh pia|mi quang|yen sao|toi den|hat dieu|tieu den|sau rieng|xoai cat|buoi da xanh|gao st|ruou sim|ca kho|banh com|cha muc|vai thieu|tra tan cuong|muc kho|bun bo|nuoc mam|ca phe|nem chua|keo dua|banh trang|banh cuon|hu tieu|bo to|banh da cua|cao lau\s+(?:hoi an|quang nam|dac san)|bun dau|banh xeo|lap xuong|banh chung|me xung|tom kho|cua bien|mat ong|nguyen hat|mot nang|loai 1|chinh hang|gia re)\b/i;

  const CHAT_NARRATIVE_RE = /\b(?:xin gia|phi ship|bao nhieu|may ngay|o dau|tinh sao|mat may|nghi le|nghi trua|cam xe|di bo|noi tieng|tang cao|to chuc|dong benh nhan|phong tro|ve que|nhan tin|truong phong|chien si|thac nuoc|mua thu|phun lua|benh vien|buu cuc|tuyen nhan vien|tuyen dung|cua hang o|chi nhanh o|chi nhanh|cong ty|truong dai hoc|gia dat|gia ve|tau cao toc|chay tuyen|du lich|thoi tiet|hom nay|ngay mai|hom qua|ket xe|danh nhan|cho minh|cho em|dat\s+(?:hang|don|mon|ship|mua|cho|giup|\d+)|ok shop|ship nhanh|xin chao|cam on|vang a|o cho cu|nhu lan truoc|nhu don lan truoc|chuyen khoan|con hang|freeship|giam gia|mua to|nang nong|mat dien|ngap nuoc|sinh nam|goi cho|ve may bay|xe khach|gio hanh chinh giup|kiem tra hang|hang de vo|bam chuong|ky nhan ho|ngoai cong|khong day|size\s*(?:s|m|l|xl|xxl|\d{2})|mau\s+(?:den|trang|xanh|do|be|xam)|chuc\s+(?:mung|suc|tet|ngu|vui|em|anh|chi|ban|shop|co|bac|nam|ngay|cai|chiec|qua|hop|goi|bich|tui|vu|nang)|(?:mui|toa|thom|sac|ngat|mua thu)\s+huong\s+hoa|huong\s+hoa\s+(?:sua|lai|sen|thom|co|nhai|buoi|bay|ngat))\b/i;

  const ACCENTED_FOOD_TRAP_RE = /(?:^|[\s,])(?:phở|mực|đặt|hộp|hương\s+hoa|cao\s+lầu|chức)(?:\s|$|,)/i;
  const ANH_HUNG_TRAP_RE = /\b(?:vi\s+anh\s+hung|anh\s+hung\s+(?:dan\s+toc|liet\s+si|luc\s+luong)|^anh\s+hung\b)\b/i;
  const DELIVERY_CONTACT_RE = /\b(?:giao|ship|gui|nhan tai|dia chi|d\/c|dc|sdt|lien he|0\d{8,9})\b/i;

  const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/i;
  const URL_RE = /^(?:https?:\/\/\S+|www\.\S+|[a-z0-9-]+\.(?:vn|com|net|org|io|gov\.vn|edu\.vn)(?:\/\S*)?)$/i;
  const PHONE_ONLY_RE = /^(?:(?:sdt|so dt|zalo|lh|lien he|tel|phone)\s*[:.]?\s*)?(?:\+?84|0)(?:[\s.\-]*\d){8,10}$/i;
  const DATE_ONLY_RE = /^(?:\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}|\d{4}[\/\-.]\d{1,2}[\/\-.]\d{1,2})$/;
  const PRICE_ONLY_RE = /^\d{1,3}(?:[.,]\d{3})*(?:\s*(?:d|đ|vnd|dong|k|tr|trieu))$/i;
  const TRACKING_ID_RE = /^(?:ghn|ghtk|spx|spxvn|vtp|jt|mvd|dh|ord|sku|stk|cccd|mst)[a-z0-9\-_]*\d{3,}$/i;
  const DIGITS_SYMBOLS_RE = /^[\d\s.,\-/:+#*()=?%_]+$/;

  const WARD_PREFIX_SET = new Set(["phuong", "xa", "thi tran", "dac khu", "p", "x", "tt", "dk", "ph"]);
  const DIST_PREFIX_SET = new Set(["quan", "huyen", "thi xa", "thanh pho", "q", "h", "tx", "tp"]);
  const PROV_PREFIX_SET = new Set(["tinh", "thanh pho", "tp", "t"]);

  const AMBIGUOUS_SINGLE_WORD_ALIASES = new Set(["vinh"]);

  const state = {
    initialized: false,
    raw: null,
    p: [],
    d: [],
    w: [],
    e: [],
    dp: [],
    provinceMap: new Map(), // string -> pid
    districtMap: new Map(), // string -> did
    wardMap: new Map(),     // string -> wid
    maxPhraseWords: 6,
  };

  function stripDiacritics(str) {
    return (str || "")
      .replace(/đ/g, "d")
      .replace(/Đ/g, "D")
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "");
  }

  function normalizeForRules(str) {
    return stripDiacritics((str || "").trim())
      .toLowerCase()
      .replace(/\s+/g, " ");
  }

  function initGazetteer(gazetteerJson) {
    if (!gazetteerJson || !Array.isArray(gazetteerJson.p)) {
      throw new Error("Invalid vn_admin_gazetteer.json payload");
    }
    state.raw = gazetteerJson;
    state.p = gazetteerJson.p;
    state.d = gazetteerJson.d;
    state.w = gazetteerJson.w;
    state.e = gazetteerJson.e;
    state.dp = gazetteerJson.dp;

    state.provinceMap = new Map();
    state.districtMap = new Map();
    state.wardMap = new Map();

    let maxWords = 1;
    const regMax = (s) => {
      const c = s.split(" ").length;
      if (c > maxWords) maxWords = c;
    };

    for (let i = 0; i < state.p.length; i++) {
      const key = state.p[i];
      state.provinceMap.set(key, i);
      regMax(key);
      // Also register hyphen-free variant for "ba ria - vung tau"
      const noHyphen = key.replace(/\s*-\s*/g, " ");
      if (noHyphen !== key) {
        state.provinceMap.set(noHyphen, i);
        regMax(noHyphen);
      }
    }

    if (gazetteerJson.pa) {
      for (const [alias, canon] of Object.entries(gazetteerJson.pa)) {
        const pid = state.provinceMap.get(canon);
        if (pid !== undefined) {
          const cleanAlias = alias.replace(/\s*-\s*/g, " ").replace(/\s+/g, " ").trim();
          state.provinceMap.set(cleanAlias, pid);
          regMax(cleanAlias);
        }
      }
    }

    for (let i = 0; i < state.d.length; i++) {
      const key = state.d[i];
      state.districtMap.set(key, i);
      regMax(key);
      const noHyphen = key.replace(/\s*-\s*/g, " ").replace(/\s+/g, " ").trim();
      if (noHyphen !== key) {
        state.districtMap.set(noHyphen, i);
        regMax(noHyphen);
      }
    }

    if (gazetteerJson.da) {
      for (const [alias, canon] of Object.entries(gazetteerJson.da)) {
        const did = state.districtMap.get(canon);
        if (did !== undefined) {
          const cleanAlias = alias.replace(/\s+/g, " ").trim();
          state.districtMap.set(cleanAlias, did);
          regMax(cleanAlias);
        }
      }
    }

    for (let i = 0; i < state.w.length; i++) {
      const key = state.w[i];
      state.wardMap.set(key, i);
      regMax(key);
      const noHyphen = key.replace(/\s*-\s*/g, " ").replace(/\s+/g, " ").trim();
      if (noHyphen !== key) {
        state.wardMap.set(noHyphen, i);
        regMax(noHyphen);
      }
    }

    state.maxPhraseWords = Math.min(8, maxWords);
    state.initialized = true;
    return true;
  }

  function ensureGazetteerLoaded() {
    if (state.initialized) return true;
    // Lazy auto-load in Node environments if vn_admin_gazetteer.json sits beside heuristics.js
    if (typeof require !== "undefined" && typeof __dirname !== "undefined") {
      try {
        const fs = require("fs");
        const path = require("path");
        const candidate = path.join(__dirname, "model", "vn_admin_gazetteer.json");
        if (fs.existsSync(candidate)) {
          const json = JSON.parse(fs.readFileSync(candidate, "utf8"));
          return initGazetteer(json);
        }
      } catch (_) {
        // Ignore and fall through cleanly
      }
    }
    return false;
  }

  /**
   * Tokenizes normalized text into clean word tokens, expanding attached VN admin
   * abbreviations (`x.loc` -> `x.`, `loc`; `p.12` -> `p.`, `12`; `q.tb` -> `q.`, `tb`).
   * Keeps punctuation boundary markers (`,`, `;`) to avoid accidental cross-comma merges
   * when useful, while tracking exact token indices.
   */
  function tokenizeForGazetteer(norm) {
    let s = norm
      .replace(/[,;|]+/g, " , ")
      .replace(/\s+-\s+/g, " ")
      .replace(/(^|\s)tp\.?\s*hcm(?=\s|$)/g, "$1tphcm")
      .replace(/(^|\s)tp\.?\s*hn(?=\s|$)/g, "$1tphn")
      .replace(/(^|\s)br\s*-\s*vt(?=\s|$)/g, "$1brvt")
      // Split attached prefix dots: x.loc -> x. loc, p.12 -> p. 12, tt.dong -> tt. dong
      .replace(/(^|\s)(x|p|q|h|t|tt|tx|tp|dk)\.(?=[a-z0-9])/g, "$1$2. ")
      // Split compact p12 / q1..q12 -> p 12 / q 1..q 12
      .replace(/(^|\s)(p|q)(\d{1,2})(?=\s|$)/g, "$1$2 $3")
      // Remove trailing dots on tokens (x. -> x, q. -> q)
      .replace(/\.(?=\s|$)/g, "")
      .replace(/\s+/g, " ")
      .trim();

    if (!s) return [];
    return s.split(" ");
  }

  function detectPrefixBefore(tokens, idx) {
    if (idx <= 0) return { kind: null, start: idx };
    const t1 = tokens[idx - 1];
    if (t1 === ",") return { kind: null, start: idx };
    if (idx >= 2 && tokens[idx - 2] !== ",") {
      const t2 = tokens[idx - 2] + " " + t1;
      if (t2 === "thi tran" || t2 === "dac khu") {
        return { kind: "ward", start: idx - 2 };
      }
      if (t2 === "thi xa" || t2 === "thanh pho") {
        return { kind: t2 === "thanh pho" ? "dist_or_prov" : "dist", start: idx - 2 };
      }
    }
    if (WARD_PREFIX_SET.has(t1)) return { kind: "ward", start: idx - 1 };
    if (DIST_PREFIX_SET.has(t1)) return { kind: t1 === "tp" ? "dist_or_prov" : "dist", start: idx - 1 };
    if (PROV_PREFIX_SET.has(t1)) return { kind: "prov", start: idx - 1 };
    return { kind: null, start: idx };
  }

  function findGazetteerSpans(tokens) {
    const wards = [];
    const districts = [];
    const provinces = [];
    const n = tokens.length;

    for (let i = 0; i < n; i++) {
      if (tokens[i] === ",") continue;
      const pfx = detectPrefixBefore(tokens, i);
      const maxLen = Math.min(state.maxPhraseWords, n - i);

      for (let len = maxLen; len >= 1; len--) {
        const slice = tokens.slice(i, i + len);
        if (slice.includes(",")) continue;
        const phrase = slice.join(" ");
        const wordCount = len;

        // 1. Ward candidate (skip if preceded by explicit district/province prefix!)
        const wid = state.wardMap.get(phrase);
        const hasNonWardPrefix = pfx.kind === "dist" || pfx.kind === "prov" || pfx.kind === "dist_or_prov";
        if (wid !== undefined && !hasNonWardPrefix) {
          const hasWardPrefix = pfx.kind === "ward";
          const isNumericOnly = /^\d+$/.test(phrase);
          if (!isNumericOnly) {
            const spanStart = hasWardPrefix ? pfx.start : i;
            wards.push({
              id: wid,
              name: phrase,
              start: spanStart,
              end: i + len,
              wordStart: i,
              wordCount,
              prefixed: hasWardPrefix,
            });
          }
        }
        // Also check prefixed numeric ward ("p 12" -> "phuong 12")
        if (len === 1 && /^\d{1,2}$/.test(phrase) && pfx.kind === "ward") {
          const numWid = state.wardMap.get("phuong " + phrase);
          if (numWid !== undefined) {
            wards.push({
              id: numWid,
              name: "phuong " + phrase,
              start: pfx.start,
              end: i + 1,
              wordStart: i,
              wordCount: 1,
              prefixed: true,
            });
          }
        }

        // 2. District candidate (always keep unprefixed [i, i+len] so a preceding ward ending in "quan"/"xa" never blocks!)
        let did = state.districtMap.get(phrase);
        if (did === undefined && len === 1 && /^\d{1,2}$/.test(phrase) && (pfx.kind === "dist" || tokens[i - 1] === "q" || tokens[i - 1] === "quan")) {
          did = state.districtMap.get("quan " + phrase);
        }
        if (did !== undefined) {
          const hasDistPrefix = pfx.kind === "dist" || pfx.kind === "dist_or_prov";
          const isAmbiguousSingle = wordCount === 1 && AMBIGUOUS_SINGLE_WORD_ALIASES.has(phrase) && !hasDistPrefix;
          if (hasDistPrefix) {
            districts.push({
              id: did,
              name: phrase,
              start: pfx.start,
              end: i + len,
              wordStart: i,
              prefixed: true,
              ambiguousSingle: false,
            });
          }
          districts.push({
            id: did,
            name: phrase,
            start: i,
            end: i + len,
            wordStart: i,
            prefixed: false,
            ambiguousSingle: isAmbiguousSingle,
          });
        }

        // 3. Province candidate (always keep unprefixed [i, i+len] too so "Hoa Xa Ha Noi" matches!)
        const pid = state.provinceMap.get(phrase);
        if (pid !== undefined) {
          const hasProvPrefix = pfx.kind === "prov" || pfx.kind === "dist_or_prov";
          const isAmbiguousSingle = wordCount === 1 && AMBIGUOUS_SINGLE_WORD_ALIASES.has(phrase) && !hasProvPrefix;
          if (hasProvPrefix) {
            provinces.push({
              id: pid,
              name: phrase,
              start: pfx.start,
              end: i + len,
              wordStart: i,
              prefixed: true,
              ambiguousSingle: false,
            });
          }
          provinces.push({
            id: pid,
            name: phrase,
            start: i,
            end: i + len,
            wordStart: i,
            prefixed: false,
            ambiguousSingle: isAmbiguousSingle,
          });
        }
      }
    }

    return { wards, districts, provinces };
  }

  function spansDisjoint(a, b) {
    return a.end <= b.start || b.end <= a.start;
  }

  function hasValidWardDistrictEdge(wid, did) {
    const edges = state.e[wid];
    if (!edges) return false;
    for (let i = 0; i < edges.length; i++) {
      if (edges[i][0] === did) return true;
    }
    return false;
  }

  function hasValidWardProvinceEdge(wid, pid) {
    const edges = state.e[wid];
    if (!edges) return false;
    for (let i = 0; i < edges.length; i++) {
      if (edges[i][1] === pid) return true;
    }
    return false;
  }

  function hasValidDistrictProvinceEdge(did, pid) {
    const pids = state.dp[did];
    if (!pids) return false;
    return pids.includes(pid);
  }

  function findBestAdminHierarchy(spans) {
    const { wards, districts, provinces } = spans;

    // 1. Try Ward + District (+ optional Province)
    let bestWD = null;
    for (const w of wards) {
      for (const d of districts) {
        if (!spansDisjoint(w, d)) continue;
        // Prefer natural left-to-right VN order: ward before district
        if (w.start > d.start) continue;
        if (!hasValidWardDistrictEdge(w.id, d.id)) continue;
        // 1-word unprefixed wards (e.g. "Bưởi, Tây Hồ") allowed ONLY when directly paired with a valid disjoint district!
        const score = (w.end - w.start) * 2 + (d.end - d.start) * 2 + (w.prefixed ? 1 : 0) + (d.prefixed ? 1 : 0);
        if (!bestWD || score > bestWD.score) {
          bestWD = { ward: w, district: d, score };
        }
      }
    }
    if (bestWD) return { type: "ward_district", ...bestWD };

    // 2. Check District + Province FIRST if a District span equals an unprefixed Ward span
    //    (e.g. "Thanh Xuân, Hà Nội" or "Lộc Ninh, Bình Phước" is District+Province, NOT Ward+Province!)
    let bestDP = null;
    for (const d of districts) {
      for (const p of provinces) {
        if (!spansDisjoint(d, p)) continue;
        if (d.start > p.start) continue;
        if (!hasValidDistrictProvinceEdge(d.id, p.id)) continue;
        const score = (d.end - d.start) * 2 + (p.end - p.start) * 2 + (d.prefixed ? 2 : 0);
        if (!bestDP || score > bestDP.score) {
          bestDP = { district: d, province: p, score };
        }
      }
    }

    // 3. Try Ward + Province (2025 2-tier or missing district)
    let bestWP = null;
    for (const w of wards) {
      // Unprefixed 1-word wards are too ambiguous against province-only (require >=2 words or prefix for ward+province)
      if (!w.prefixed && w.wordCount < 2) continue;
      for (const p of provinces) {
        if (!spansDisjoint(w, p)) continue;
        if (w.start > p.start) continue;
        if (!hasValidWardProvinceEdge(w.id, p.id)) continue;
        // If unprefixed ward span overlaps/matches a valid District+Province span, prefer District+Province!
        const shadowedByDistrict =
          !w.prefixed &&
          bestDP &&
          !(w.end <= bestDP.district.start || bestDP.district.end <= w.start);
        if (shadowedByDistrict) continue;
        const score = (w.end - w.start) * 2 + (p.end - p.start) * 2 + (w.prefixed ? 2 : 0) + (p.prefixed ? 1 : 0);
        if (!bestWP || score > bestWP.score) {
          bestWP = { ward: w, province: p, score };
        }
      }
    }
    if (bestWP) return { type: "ward_province", ...bestWP };
    if (bestDP) return { type: "district_province", ...bestDP };

    return null;
  }

  function isVietnamesePersonNameTrap(norm, rawTrimmed) {
    if (rawTrimmed.includes(",") || /\d/.test(norm)) return false;
    if (/\b(?:x\.|p\.|q\.|h\.|tt\.|tx\.|tp\.|xa|phuong|quan|huyen|thi xa|thi tran|tinh|ngo|hem|ngach|kiet|duong|pho|thon|ap|xom|to|chung cu)\b/.test(norm)) {
      return false;
    }
    const words = norm ? norm.split(" ") : [];
    if (words.length < 2 || words.length > 5) return false;
    if (HISTORICAL_PERSON_TRAPS.has(norm)) return true;
    if (words.length >= 3 && SURNAMES_RE.test(words[0]) && MIDDLE_NAMES_RE.test(words[1])) {
      return true;
    }
    return false;
  }

  function extractStructuralSignals(rawText, norm) {
    const rawLow = (rawText || "").toLowerCase();
    const isAccentedFoodTrap = ACCENTED_FOOD_TRAP_RE.test(rawLow);
    const isPhoFood = /(?:^|[\s,])phở(?:\s|$|,)/i.test(rawLow);
    // Prevent "thanh pho <name>" from falsely triggering street keyword "pho <name>"
    const normNoThanhPho = norm.replace(/\bthanh\s+pho\b/g, "tp");
    const hasAlleyKw = /\b(?:ngo|ngach|hem|kiet)\s+\d{1,4}(?:\/\d{1,4}[a-z]?)?/.test(norm);
    const hasStreetKw = !isPhoFood && /\b(?:duong|pho|quoc lo|ql\s*\d+|tinh lo|dai lo|xa lo)\s+[a-z]{2,}/.test(normNoThanhPho);
    const hasHousePrefix = /\b(?:so nha|sn|so)\s+\d{1,4}(?:[a-z]|\/\d{1,3}[a-z]?){0,2}\s+[a-z]{2,}/.test(norm);
    // Strip product-unit tokens (e.g. "500g", "2 goi", "3 chai") so they never mimic house numbers
    const normNoProductUnits = norm.replace(/(?<![/.\d])\b\d+\s*(?:g|gr|kg|ml|lit|cai|chiec|goi|hop|bich|tui|chai|hu|thung|suat|vien|cuon|quyen|cm|mm|gb|tb|mah|k|tr|trieu|dong|vnd)\b/gi, " ");
    const hasHouseNumberStreet = /(?:^|[\s,:])\d{1,4}(?:[a-z]|\/\d{1,3}[a-z]?){0,2}\s+(?:[a-z]{2,}\s+){1,}[a-z]{2,}/.test(normNoProductUnits);
    const hasBuildingUnit = /\b(?:chung cu|toa nha|can ho|kdt|khu do thi|kcn|khu cong nghiep|kcx)\s+[a-z0-9]{1,}/.test(norm);
    const hasBareGranularKw = /\b(?:ngo|ngach|hem|kiet|thon|ap|xom|ban|khom|to|to dan pho|khu pho|kp|tang|lau|phong|can ho|chung cu|toa nha|block|lo)\s+(?:[a-z0-9]{1,})/i.test(norm);
    const hasDeliveryWrap = DELIVERY_CONTACT_RE.test(norm);

    const hasExplicitGranular =
      hasAlleyKw ||
      hasStreetKw ||
      hasHousePrefix ||
      hasHouseNumberStreet ||
      hasBuildingUnit;

    const hasExplicitWardPrefix = /(?:^|[\s,])(?:phuong|xa|thi tran|dac khu)\s+[a-z0-9]+|(?:^|[\s,])(?:p|x|tt|dk)\.\s*[a-z0-9]+|(?:^|[\s,])p\d{1,2}(?=\s|,|$)/.test(norm);
    const hasExplicitDistPrefix = /(?:^|[\s,])(?:quan|huyen|thi xa)\s+[a-z0-9]+|(?:^|[\s,])(?:q|h|tx)\.\s*[a-z0-9]+|(?:^|[\s,])q\d{1,2}(?=\s|,|$)/.test(norm);
    const hasExplicitProvPrefix = /(?:^|[\s,])(?:tinh|thanh pho)\s+[a-z]+|(?:^|[\s,])(?:tp|t)\.\s*[a-z]+|(?:^|[\s,])tp\s+[a-z]+/.test(norm);

    const hasAntiCue =
      isAccentedFoodTrap ||
      PRODUCT_UNITS_RE.test(norm) ||
      SPECIALTY_FOOD_PRODUCT_RE.test(norm) ||
      CHAT_NARRATIVE_RE.test(norm) ||
      /\bmuc\s+(?:ca mau|kho|mot nang|nuong|hap|xao)\b/i.test(norm) ||
      /\bpho\s+(?:bo|ga|cuon|ha noi|sai gon|nam dinh|xao|tai|chin|dac biet)\b/i.test(normNoThanhPho) ||
      /\b(?:an|quan|to|bat|banh|ship|dat|cho|mua)\s+pho\b/i.test(normNoThanhPho) ||
      (!hasDeliveryWrap && ANH_HUNG_TRAP_RE.test(norm));

    return {
      hasAlleyKw,
      hasStreetKw,
      hasHousePrefix,
      hasHouseNumberStreet,
      hasBuildingUnit,
      hasBareGranularKw,
      hasExplicitGranular,
      hasExplicitWardPrefix,
      hasExplicitDistPrefix,
      hasExplicitProvPrefix,
      hasAntiCue,
      hasDeliveryWrap,
    };
  }

  /**
   * Tier 1 Evaluation:
   * Returns {
   *   decision: 'clear_sufficient' | 'clear_insufficient' | 'not_address' | 'ambiguous',
   *   sufficiency: 'address_sufficient' | 'address_insufficient' | 'not_address' | null,
   *   is_address: boolean | null,
   *   probability: number | null,
   *   sufficiency_probs: number[] | null,
   *   reason: string,
   *   tier: string,
   *   latency_ms: number
   * }
   */
  function evaluateTier1(rawText) {
    const t0 = typeof performance !== "undefined" ? performance.now() : Date.now();
    ensureGazetteerLoaded();

    const trimmed = (rawText || "").trim();
    const norm = normalizeForRules(trimmed);
    const words = norm ? norm.split(" ") : [];

    const finish = (decision, sufficiency, isAddress, probability, sufProbs, reason) => ({
      decision,
      sufficiency,
      is_address: isAddress,
      probability,
      sufficiency_probs: sufProbs,
      reason,
      tier: decision === "ambiguous" ? "Tier 1 -> Tier 2" : "Tier 1 (Gazetteer/Rule)",
      latency_ms: +(((typeof performance !== "undefined" ? performance.now() : Date.now()) - t0)).toFixed(3),
    });

    // --- 1. Structural Non-Address Fast Reject ---
    if (!trimmed) {
      return finish("not_address", "not_address", false, 0.001, [0.996, 0.003, 0.001], "Chuỗi rỗng");
    }
    if (trimmed.length < 3) {
      return finish("not_address", "not_address", false, 0.002, [0.994, 0.004, 0.002], "Quá ngắn (< 3 ký tự)");
    }
    if (EMAIL_RE.test(trimmed)) {
      return finish("not_address", "not_address", false, 0.002, [0.995, 0.003, 0.002], "Định dạng Email");
    }
    if (URL_RE.test(trimmed)) {
      return finish("not_address", "not_address", false, 0.002, [0.995, 0.003, 0.002], "Định dạng URL / Link");
    }
    if (PHONE_ONLY_RE.test(norm)) {
      return finish("not_address", "not_address", false, 0.003, [0.993, 0.004, 0.003], "Số điện thoại đơn lẻ");
    }
    if (DATE_ONLY_RE.test(trimmed)) {
      return finish("not_address", "not_address", false, 0.003, [0.993, 0.004, 0.003], "Định dạng Ngày tháng");
    }
    if (PRICE_ONLY_RE.test(norm)) {
      return finish("not_address", "not_address", false, 0.003, [0.993, 0.004, 0.003], "Định dạng Giá tiền");
    }
    if (TRACKING_ID_RE.test(norm)) {
      return finish("not_address", "not_address", false, 0.003, [0.992, 0.005, 0.003], "Mã vận đơn / Mã định danh");
    }
    if (DIGITS_SYMBOLS_RE.test(trimmed)) {
      return finish("not_address", "not_address", false, 0.002, [0.994, 0.004, 0.002], "Chỉ chứa số / ký hiệu");
    }

    // --- 2. Person Name Guard (before admin unit matching!) ---
    if (isVietnamesePersonNameTrap(norm, trimmed)) {
      return finish(
        "not_address",
        "not_address",
        false,
        0.008,
        [0.975, 0.017, 0.008],
        "Họ tên người Việt / Danh nhân (không phải địa chỉ)"
      );
    }

    // --- 3. Gazetteer Span Matching & Structural Signals ---
    const sig = extractStructuralSignals(trimmed, norm);
    const tokens = tokenizeForGazetteer(norm);
    const spans = state.initialized
      ? findGazetteerSpans(tokens)
      : { wards: [], districts: [], provinces: [] };
    const multiWordOrPrefixedWards = spans.wards.filter((w) => w.prefixed || w.wordCount >= 2);

    // Single-token check: reject unless it cleanly matches an explicit/canonical admin unit (e.g. "hcm", "tphcm", "p.12")
    const clearDistricts = spans.districts.filter((d) => !d.ambiguousSingle);
    const clearProvinces = spans.provinces.filter((p) => !p.ambiguousSingle);
    if (words.length === 1 && !sig.hasExplicitWardPrefix && !sig.hasExplicitDistPrefix) {
      const hasSingleAdmin = clearProvinces.length > 0 || clearDistricts.length > 0;
      if (!hasSingleAdmin) {
        return finish("not_address", "not_address", false, 0.005, [0.985, 0.010, 0.005], "Từ đơn không khớp đơn vị hành chính");
      }
    }

    const hierarchy = state.initialized ? findBestAdminHierarchy(spans) : null;
    const hasWardParent = hierarchy && (hierarchy.type === "ward_district" || hierarchy.type === "ward_province");
    // Only count district/province spans that are NOT identical to a prefixed ward span
    const primaryWard = multiWordOrPrefixedWards[0];
    const hasDisjointDistOrProv =
      clearDistricts.some((d) => !primaryWard || spansDisjoint(d, primaryWard) || !primaryWard.prefixed) ||
      clearProvinces.some((p) => !primaryWard || spansDisjoint(p, primaryWard) || !primaryWard.prefixed);
    const hasDistrictOrProv = clearDistricts.length > 0 || clearProvinces.length > 0;
    // Guard against "pho ha noi" (no street number) false matching street + province
    const hasGranularWithParent =
      sig.hasExplicitGranular &&
      hasDistrictOrProv &&
      !( !sig.hasHouseNumberStreet && !sig.hasHousePrefix && !sig.hasAlleyKw && !sig.hasBuildingUnit && /^pho\s+/.test(norm) );

    // --- 4. Anti-Cue / Customer Chat / Food-Place Trap Guard ---
    if (sig.hasAntiCue) {
      if (sig.hasDeliveryWrap && sig.hasExplicitGranular) {
        return finish(
          "ambiguous",
          null,
          null,
          null,
          null,
          "Chuyển Tier 2 ONNX (có đồng thời ngữ cảnh giao hàng và số nhà/đường/ngõ)"
        );
      }
      return finish(
        "not_address",
        "not_address",
        false,
        0.010,
        [0.970, 0.020, 0.010],
        "Cụm từ đơn hàng / món ăn / hội thoại không phải địa chỉ giao hàng"
      );
    }

    // --- 4b. Fast-Reject Generic Non-Address Phrases (no commas, no digits, <=8 words, no admin spans/prefixes/granular kw) ---
    if (
      !trimmed.includes(",") &&
      !/\d/.test(norm) &&
      words.length <= 8 &&
      !sig.hasExplicitWardPrefix &&
      !sig.hasExplicitDistPrefix &&
      !sig.hasExplicitProvPrefix &&
      !sig.hasExplicitGranular &&
      !sig.hasBareGranularKw &&
      multiWordOrPrefixedWards.length === 0 &&
      clearDistricts.length === 0 &&
      clearProvinces.length === 0
    ) {
      return finish(
        "not_address",
        "not_address",
        false,
        0.008,
        [0.976, 0.016, 0.008],
        "Cụm từ hội thoại / sản phẩm chung không chứa thành phần địa chỉ VN"
      );
    }

    // --- 5. Fast-Accept Clear Sufficient VN Addresses ---
    if (hasWardParent) {
      const desc =
        hierarchy.type === "ward_district"
          ? `Khớp cấp Phường/Xã + Quận/Huyện (${hierarchy.ward.name}, ${hierarchy.district.name})`
          : `Khớp cấp Phường/Xã + Tỉnh/TP (${hierarchy.ward.name}, ${hierarchy.province.name})`;
      return finish("clear_sufficient", "address_sufficient", true, 0.988, [0.004, 0.008, 0.988], desc);
    }

    if (hasGranularWithParent) {
      return finish(
        "clear_sufficient",
        "address_sufficient",
        true,
        0.985,
        [0.005, 0.010, 0.985],
        "Đủ chi tiết (Số nhà/Ngõ/Đường + Quận/Tỉnh)"
      );
    }

    // --- 6. Fast-Classify Clear Insufficient VN Address Fragments ---
    if (sig.hasExplicitWardPrefix && !hasDisjointDistOrProv) {
      return finish(
        "clear_insufficient",
        "address_insufficient",
        false,
        0.030,
        [0.040, 0.930, 0.030],
        "Chỉ có Phường/Xã/Thị trấn, thiếu Quận/Huyện/Tỉnh"
      );
    }

    if (hierarchy && hierarchy.type === "district_province" && !sig.hasExplicitGranular && !sig.hasBareGranularKw) {
      return finish(
        "clear_insufficient",
        "address_insufficient",
        false,
        0.035,
        [0.035, 0.930, 0.035],
        `Thiếu Phường/Xã hoặc Số nhà/Đường (chỉ có ${hierarchy.district.name}, ${hierarchy.province.name})`
      );
    }

    if (!sig.hasExplicitGranular && !sig.hasBareGranularKw && words.length <= 5 && multiWordOrPrefixedWards.length === 0) {
      if (clearDistricts.length > 0 || clearProvinces.length > 0) {
        const label =
          clearDistricts.length > 0
            ? `Chỉ có tên Quận/Huyện/TP (${clearDistricts[0].name}), chưa đủ giao hàng`
            : `Chỉ có tên Tỉnh/Thành phố (${clearProvinces[0].name}), chưa đủ giao hàng`;
        return finish(
          "clear_insufficient",
          "address_insufficient",
          false,
          0.025,
          [0.045, 0.930, 0.025],
          label
        );
      }
    }

    if ((sig.hasBareGranularKw || sig.hasExplicitGranular) && !hasDistrictOrProv) {
      return finish(
        "clear_insufficient",
        "address_insufficient",
        false,
        0.030,
        [0.040, 0.930, 0.030],
        "Mảnh địa chỉ thô (Ngõ/Thôn/Ấp/Đường/Chung cư) thiếu Xã/Quận/Tỉnh"
      );
    }

    // --- 7. Ambiguous -> Route to Tier-2 ONNX ---
    return finish(
      "ambiguous",
      null,
      null,
      null,
      null,
      "Chuyển Tier 2 ONNX (mẫu mơ hồ / viết tắt đặc biệt / ngữ cảnh tự do)"
    );
  }

  /**
   * Gazetteer-Only Baseline (for mode comparison when ONNX is bypassed or offline).
   */
  function evaluateHeuristicOnly(rawText) {
    const t0 = typeof performance !== "undefined" ? performance.now() : Date.now();
    const t1 = evaluateTier1(rawText);
    if (t1.decision !== "ambiguous") {
      return {
        ...t1,
        tier: "Gazetteer/Rule Only",
        latency_ms: +(((typeof performance !== "undefined" ? performance.now() : Date.now()) - t0)).toFixed(3),
      };
    }

    const norm = normalizeForRules(rawText);
    const sig = extractStructuralSignals(rawText, norm);
    const tokens = tokenizeForGazetteer(norm);
    const spans = state.initialized ? findGazetteerSpans(tokens) : { wards: [], districts: [], provinces: [] };

    if (spans.wards.length > 0 && !sig.hasAntiCue) {
      return {
        decision: "clear_insufficient",
        sufficiency: "address_insufficient",
        is_address: false,
        probability: 0.04,
        sufficiency_probs: [0.06, 0.90, 0.04],
        reason: "Gazetteer: chỉ nhận diện tên Phường/Xã đơn lẻ",
        tier: "Gazetteer/Rule Only",
        latency_ms: +(((typeof performance !== "undefined" ? performance.now() : Date.now()) - t0)).toFixed(3),
      };
    }

    return {
      decision: "not_address",
      sufficiency: "not_address",
      is_address: false,
      probability: 0.05,
      sufficiency_probs: [0.88, 0.07, 0.05],
      reason: "Gazetteer: không đủ cấu trúc địa chỉ hợp lệ",
      tier: "Gazetteer/Rule Only",
      latency_ms: +(((typeof performance !== "undefined" ? performance.now() : Date.now()) - t0)).toFixed(3),
    };
  }

  const api = {
    initGazetteer,
    stripDiacritics,
    normalizeForRules,
    tokenizeForGazetteer,
    evaluateTier1,
    evaluateHeuristicOnly,
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    root.LayaHeuristics = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
