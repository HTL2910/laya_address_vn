/**
 * Tier 1: Fast Vietnamese Address Heuristic & Gazetteer Pre-Filter (< 1 ms)
 * Catches obvious non-addresses and high-confidence formal VN addresses;
 * routes ambiguous strings (short, abbreviated, delivery notes, tricky negatives) to Tier 2 ONNX.
 */
(function (root) {
  "use strict";

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

  // Major VN provinces / cities (normalized ASCII lowercase)
  const PROVINCES = [
    "ha noi", "ho chi minh", "tp hcm", "tp.hcm", "tphcm", "sai gon",
    "da nang", "hai phong", "can tho", "an giang", "ba ria", "vung tau",
    "bac giang", "bac kan", "bac lieu", "bac ninh", "ben tre", "binh dinh",
    "binh duong", "binh phuoc", "binh thuan", "ca mau", "cao bang",
    "dak lak", "dak nong", "dien bien", "dong nai", "dong thap", "gia lai",
    "ha giang", "ha nam", "ha tinh", "hai duong", "hau giang", "hoa binh",
    "hung yen", "khanh hoa", "kien giang", "kon tum", "lai chau", "lam dong",
    "lang son", "lao cai", "long an", "nam dinh", "nghe an", "ninh binh",
    "ninh thuan", "phu tho", "phu yen", "quang binh", "quang nam", "quang ngai",
    "quang ninh", "quang tri", "soc trang", "son la", "tay ninh", "thai binh",
    "thai nguyen", "thanh hoa", "thua thien hue", "tien giang", "tra vinh",
    "tuyen quang", "vinh long", "vinh phuc", "yen bai", "nha trang", "da lat",
    "bien hoa", "thu dau mot", "di an", "thuan an", "ha long", "quy nhon",
    "buon ma thuot", "phan thiet", "phu quoc", "my tho", "long xuyen"
  ];

  const DISTRICTS = [
    "hoan kiem", "ba dinh", "dong da", "hai ba trung", "cau giay",
    "thanh xuan", "hoang mai", "long bien", "tay ho", "nam tu liem",
    "bac tu liem", "ha dong", "dong anh", "gia lam", "thanh tri",
    "hoai duc", "thach that", "soc son", "son tay",
    "binh thanh", "tan binh", "tan phu", "phu nhuan", "go vap",
    "binh tan", "thu duc", "binh chanh", "hoc mon", "nha be", "cu chi",
    "hai chau", "thanh khe", "son tra", "ngu hanh son", "lien chieu", "cam le",
    "ngo quyen", "le chan", "hong bang", "hai an", "ninh kieu", "cai rang", "binh thuy"
  ];

  // Conversational / e-commerce / non-address cues that block Tier 1 fast-accept
  const CHAT_OR_PRODUCT_RE = /\b(hom nay|ngay mai|hom qua|mua to|ket xe|du lich|sinh nam|goi cho|mat dien|thoi tiet|luong|tuyen nhan vien|ve may bay|nang nong|ngap nuoc|an pho|quan ca phe|size\s*(s|m|l|xl|xxl|\d{2})|mau den|mau trang|mau xanh|pro max|\d+\s*(gb|tb|ml|kg|cm)|freeship|gia bao nhieu|con hang|chuyen khoan|kiem tra hang|hang de vo|khong giao|gio hanh chinh giup|bam chuong)\b/i;

  const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/i;
  const URL_RE = /^(https?:\/\/\S+|www\.\S+|[a-z0-9-]+\.(vn|com|net|org|io|gov\.vn|edu\.vn)(\/\S*)?)$/i;
  const PHONE_ONLY_RE = /^(?:(?:sdt|so dt|zalo|lh|lien he|tel|phone)\s*[:.]?\s*)?(?:\+?84|0)(?:[\s.\-]*\d){8,10}$/i;
  const DATE_ONLY_RE = /^(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}|\d{4}[\/\-.]\d{1,2}[\/\-.]\d{1,2})$/;
  const PRICE_ONLY_RE = /^\d{1,3}(?:[.,]\d{3})*(?:\s*(?:d|đ|vnd|dong|k|tr|trieu))$/i;
  const DIGITS_SYMBOLS_RE = /^[\d\s.,\-/:+#*()=?%_]+$/;

  function containsAnyPhrase(norm, list) {
    for (let i = 0; i < list.length; i++) {
      const p = list[i];
      const idx = norm.indexOf(p);
      if (idx !== -1) {
        const beforeOk = idx === 0 || !/[a-z0-9]/.test(norm[idx - 1]);
        const end = idx + p.length;
        const afterOk = end === norm.length || !/[a-z0-9]/.test(norm[end]);
        if (beforeOk && afterOk) return p;
      }
    }
    return null;
  }

  function extractSignals(rawText) {
    const trimmed = (rawText || "").trim();
    const norm = normalizeForRules(trimmed);

    const matchedProvince = containsAnyPhrase(norm, PROVINCES);
    const matchedCityAbbr = /\b(hcm|hn|sg|dn|hp)\b/.test(norm);
    const hasProvince = Boolean(matchedProvince || matchedCityAbbr);

    const matchedDistrict = containsAnyPhrase(norm, DISTRICTS);
    const hasExplicitDistrict = /\b(quan|huyen|thi xa|tp\.?\s*thu duc|q\.?\s*\d{1,2}|q\.\s*[a-z]{2,}|q(?:tb|bt|pn|gv|tp|tx|cg|dd|hk))\b/.test(norm);
    const hasDistrict = Boolean(matchedDistrict || hasExplicitDistrict);

    const hasWard = /\b(phuong|xa|thi tran|tt\.|p\.?\s*\d{1,2}|p\.\s*[a-z]{2,})\b/.test(norm);
    const hasAlleyOrStreetKw = /\b(so nha|so\s+\d+|sn\s+\d+|ngo\s+\d+|ngach\s+\d+|hem\s+\d+|kiet\s+\d+|duong\s+[a-z]|pho\s+[a-z]|quoc lo|ql\s*\d+|tinh lo|dai lo|xa lo)\b/.test(norm);
    const hasBuildingOrRural = /\b(chung cu|toa nha|can ho|tang\s+\d+|lau\s+\d+|block\s+[a-z0-9]|lo\s+[a-z0-9]|kcn|khu cong nghiep|kcx|khu do thi|kdt|thon\s+|ap\s+|xom\s+|to dan pho|khu pho|kp\.?\s*\d+)\b/.test(norm);
    const hasHouseNumber = /(?:^|[\s,:])(?:so\s+|sn\s+)?\d{1,4}(?:[a-z]|\/\d{1,3}[a-z]?){0,2}\s+[a-z]{2,}/.test(norm);
    const hasComma = trimmed.includes(",");
    const hasAntiCue = CHAT_OR_PRODUCT_RE.test(norm);
    const wordCount = norm ? norm.split(" ").length : 0;

    return {
      trimmed,
      norm,
      wordCount,
      matchedProvince,
      matchedDistrict,
      hasProvince,
      hasDistrict,
      hasExplicitDistrict,
      hasWard,
      hasAlleyOrStreetKw,
      hasBuildingOrRural,
      hasHouseNumber,
      hasComma,
      hasAntiCue,
    };
  }

  /**
   * Tier 1 Pre-Filter (used by Hybrid 2-Tier mode).
   * Returns { decision: 'address' | 'not_address' | 'ambiguous', is_address: boolean|null, probability: number|null, reason: string, latency_ms: number }
   */
  function evaluateTier1(rawText) {
    const t0 = performance.now();
    const s = extractSignals(rawText);

    const finish = (decision, isAddress, probability, reason) => ({
      decision,
      is_address: isAddress,
      probability,
      reason,
      tier: "Tier 1 (Heuristic)",
      latency_ms: +(performance.now() - t0).toFixed(3),
    });

    // --- Fast-Reject Obvious Garbage ---
    if (s.trimmed.length === 0) {
      return finish("not_address", false, 0.001, "Chuỗi rỗng");
    }
    if (s.trimmed.length < 5) {
      return finish("not_address", false, 0.004, "Quá ngắn (< 5 ký tự)");
    }
    if (EMAIL_RE.test(s.trimmed)) {
      return finish("not_address", false, 0.002, "Định dạng Email");
    }
    if (URL_RE.test(s.trimmed)) {
      return finish("not_address", false, 0.002, "Định dạng URL / Link");
    }
    if (PHONE_ONLY_RE.test(s.norm)) {
      return finish("not_address", false, 0.003, "Định dạng Số điện thoại");
    }
    if (DATE_ONLY_RE.test(s.trimmed)) {
      return finish("not_address", false, 0.004, "Định dạng Ngày tháng");
    }
    if (PRICE_ONLY_RE.test(s.norm)) {
      return finish("not_address", false, 0.004, "Định dạng Giá tiền");
    }
    if (DIGITS_SYMBOLS_RE.test(s.trimmed)) {
      return finish("not_address", false, 0.003, "Chỉ chứa số / ký hiệu");
    }
    if (s.wordCount === 1 && !s.hasWard && !s.hasExplicitDistrict) {
      return finish("not_address", false, 0.008, "Từ đơn không có cấu trúc địa chỉ");
    }

    // --- Fast-Accept High-Confidence Formal VN Addresses ---
    if (!s.hasAntiCue) {
      const strongAdminCount =
        (s.hasWard ? 1 : 0) +
        (s.hasDistrict ? 1 : 0) +
        (s.hasProvince ? 1 : 0);

      // Formal full address: explicit street/alley/house + ward + district/province + comma
      if ((s.hasAlleyOrStreetKw || s.hasHouseNumber) && s.hasWard && (s.hasDistrict || s.hasProvince) && s.hasComma) {
        return finish("address", true, 0.992, "Địa chỉ hành chính đầy đủ (Số/Đường + Phường/Xã + Quận/Tỉnh)");
      }
      // Building / Industrial Zone / Rural Hamlet with ward or district + province
      if (s.hasBuildingOrRural && strongAdminCount >= 2 && s.hasComma) {
        return finish("address", true, 0.988, "Địa chỉ Tòa nhà / KCN / Thôn-Ấp kèm địa giới hành chính");
      }
      // Explicit 3-tier administrative structure with formal prefixes (Phường/Xã + Quận/Huyện + Tỉnh/TP)
      if (s.hasWard && s.hasExplicitDistrict && s.matchedProvince && s.hasComma) {
        return finish("address", true, 0.982, "Đủ 3 cấp hành chính VN (Phường/Xã, Quận/Huyện, Tỉnh/TP)");
      }
    }

    // --- Ambiguous -> Route to Tier 2 ONNX Noul Model ---
    return {
      decision: "ambiguous",
      is_address: null,
      probability: null,
      reason: "Chuyển Tier 2 ONNX (chuỗi ngắn / viết tắt / ngữ cảnh cần mô hình xử lý)",
      tier: "Tier 1 -> Tier 2",
      latency_ms: +(performance.now() - t0).toFixed(3),
    };
  }

  /**
   * Pure Heuristic-Only classifier (for A/B comparison against Hybrid and Pure ONNX).
   */
  function evaluateHeuristicOnly(rawText) {
    const t0 = performance.now();
    const t1 = evaluateTier1(rawText);
    if (t1.decision !== "ambiguous") {
      return {
        ...t1,
        tier: "Heuristic Only",
        latency_ms: +(performance.now() - t0).toFixed(3),
      };
    }

    const s = extractSignals(rawText);
    let score = 0;
    const reasons = [];

    if (s.hasAlleyOrStreetKw) { score += 2.2; reasons.push("từ khóa đường/ngõ"); }
    if (s.hasBuildingOrRural) { score += 2.2; reasons.push("tòa nhà/KCN/thôn-ấp"); }
    if (s.hasWard) { score += 1.6; reasons.push("phường/xã"); }
    if (s.hasDistrict) { score += 1.4; reasons.push("quận/huyện"); }
    if (s.hasProvince) { score += 1.1; reasons.push("tỉnh/thành"); }
    if (s.hasHouseNumber) { score += 1.0; reasons.push("số nhà"); }
    if (s.hasComma) { score += 0.4; }
    if (s.hasAntiCue) { score -= 3.5; reasons.push("từ ngữ hội thoại/sản phẩm"); }

    // Map heuristic score to pseudo-probability via logistic function
    const prob = 1 / (1 + Math.exp(-(score - 2.4)));
    const isAddress = prob >= 0.5;
    return {
      decision: isAddress ? "address" : "not_address",
      is_address: isAddress,
      probability: +prob.toFixed(4),
      reason: reasons.length
        ? `Luật điểm=${score.toFixed(1)} (${reasons.join(", ")})`
        : "Không khớp quy tắc địa chỉ VN",
      tier: "Heuristic Only",
      latency_ms: +(performance.now() - t0).toFixed(3),
    };
  }

  const api = {
    stripDiacritics,
    normalizeForRules,
    evaluateTier1,
    evaluateHeuristicOnly,
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    root.LayaHeuristics = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
