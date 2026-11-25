/**
 * Laya Address VN v3 — Vietnamese Telex/VNI Typo Repair, 2025 2-Tier Converter,
 * SymSpell-style 1-Edit Gazetteer Fuzzy Recovery, and Top-3 Ward Suggestions.
 *
 * UMD module: exports `window.LayaTelexFuzzy` in browsers and `module.exports` in Node.js.
 * MIT Licensed administrative data from tranngocminhhieu/vietnamadminunits.
 */
(function (root) {
  "use strict";

  const WARD_TYPES = ["Phường", "Xã", "Thị trấn", "Đặc khu"];

  const CENTRALLY_GOVERNED_2025 = new Set([
    "ha noi",
    "ho chi minh",
    "hai phong",
    "da nang",
    "can tho",
    "hue",
  ]);

  // Legacy urban districts that take "Quận " rather than "Huyện " when unprefixed
  const LEGACY_QUAN_NAMES = new Set([
    "ba dinh", "hoan kiem", "tay ho", "long bien", "cau giay", "dong da",
    "hai ba trung", "hoang mai", "thanh xuan", "ha dong", "nam tu liem", "bac tu liem",
    "tan binh", "binh thanh", "phu nhuan", "go vap", "tan phu", "binh tan",
    "hong bang", "ngo quyen", "le chan", "hai an", "kien an", "do son", "duong kinh", "an duong",
    "hai chau", "thanh khe", "son tra", "ngu hanh son", "lien chieu", "cam le",
    "ninh kieu", "binh thuy", "cai rang", "o mon", "thot not",
    "thuan hoa", "phu xuan",
  ]);

  const LEGACY_TP_NAMES = new Set([
    "thu duc", "thuy nguyen", "bien hoa", "long khanh", "thu dau mot", "thuan an", "di an",
    "tan uyen", "ben cat", "vung tau", "ba ria", "phu my", "da lat", "bao loc", "nha trang",
    "cam ranh", "buon ma thuot", "pleiku", "quy nhon", "quang ngai", "tam ky", "hoi an",
    "hue", "dong hoi", "dong ha", "vinh", "ha tinh", "thanh hoa", "sam son", "ninh binh",
    "tam diep", "hoa lu", "nam dinh", "thai binh", "hung yen", "hai duong", "chi linh",
    "bac ninh", "tu son", "bac giang", "thai nguyen", "song cong", "pho yen", "viet tri",
    "vinh yen", "phuc yen", "ha long", "cam pha", "uong bi", "mong cai", "dong trieu",
    "lang son", "cao bang", "bac kan", "tuyen quang", "ha giang", "lao cai", "yen bai",
    "son la", "dien bien phu", "lai chau", "hoa binh", "phan thiet", "phan rang - thap cham",
    "gia nghia", "kon tum", "tuy hoa", "dong xoai", "tay ninh", "tan an", "my tho",
    "ben tre", "tra vinh", "vinh long", "sa dec", "cao lanh", "long xuyen", "chau doc",
    "rach gia", "ha tien", "phu quoc", "vi thanh", "nga bay", "soc trang", "bac lieu", "ca mau",
  ]);

  const ENGLISH_PROTECTED_WORDS = new Set([
    "ok", "okay", "shop", "shops", "jean", "jeans", "bbq", "size", "sizes",
    "free", "freeship", "ship", "ships", "food", "foods", "good", "cool", "room",
    "book", "books", "look", "door", "doors", "floor", "poor", "moon", "noon", "soon",
    "boom", "zoom", "loop", "boot", "boots", "root", "foot", "wood", "wool", "hood",
    "mood", "pool", "tool", "school", "green", "street", "tree", "trees", "three",
    "see", "bee", "tee", "deep", "keep", "sleep", "meet", "week", "need", "feed",
    "seed", "speed", "steel", "sweet", "wheel", "sheet", "feet", "coffee", "tea",
    "milk", "ice", "cream", "cake", "cakes", "snack", "snacks", "pizza", "burger",
    "sushi", "steak", "salad", "juice", "soda", "coke", "pepsi", "beer", "wine",
    "beef", "chef", "bar", "bars", "car", "cars", "box", "boxes", "max", "mix",
    "fix", "six", "tax", "fax", "wax", "flex", "index", "complex", "duplex",
    "bus", "gas", "has", "his", "its", "yes", "pass", "plus", "sale", "off",
    "staff", "self", "half", "roof", "hair", "air", "pair", "fair", "chair", "stair",
    "year", "near", "dear", "clear", "bear", "wear", "pear", "hear", "fear", "gear",
    "tear", "her", "sir", "for", "per", "order", "orders", "tower", "towers",
    "center", "centre", "corner", "border", "river", "silver", "flower", "shower",
    "lower", "owner", "super", "water", "paper", "number", "letter", "better",
    "summer", "winter", "dinner", "butter", "sugar", "pepper", "user", "users",
    "how", "now", "low", "cow", "bow", "row", "show", "flow", "slow", "grow",
    "know", "snow", "down", "town", "brown", "crown", "raw", "law", "saw", "draw",
    "new", "few", "view", "review", "follow", "allow", "yellow", "window", "shadow",
    "narrow", "borrow", "tomorrow", "pro", "vip", "mini", "combo", "set", "full",
    "hot", "top", "tops", "best", "real", "nice", "fine", "true", "false",
    "hello", "hi", "bye", "thanks", "thank", "please", "sorry", "check", "test",
    "demo", "chat", "call", "inbox", "bill", "cash", "bank", "card", "code", "pin",
    "otp", "sim", "wifi", "app", "web", "link", "url", "email", "mail", "gmail",
    "yahoo", "zalo", "facebook", "fb", "tiktok", "shopee", "lazada", "tiki",
    "grab", "gojek", "nike", "adidas", "puma", "zara", "uniqlo", "gucci", "dior",
    "iphone", "ipad", "macbook", "samsung", "xiaomi", "oppo", "sony", "dell",
    "asus", "acer", "lenovo", "ddos", "admin", "boss", "client", "item", "items",
    "shirt", "shirts", "pants", "shoes", "bag", "bags", "hat", "hats", "cap", "caps",
    "cat", "cats", "dog", "dogs", "cup", "cups", "fan", "fans", "van", "vans",
    "can", "cans", "pan", "pans", "xl", "xxl", "xs", "ml", "kg", "gr", "cm", "mm",
    "gb", "tb", "mah", "vnd", "usd", "mart", "store", "mall", "plaza", "studio",
    "clinic", "salon", "spa", "gym", "club", "pub", "home", "house", "land", "park",
    "city", "hotel", "resort", "villa", "office", "block", "floor", "lobby", "gate",
    "grill", "hotpot", "drink", "drinks", "menu", "bill", "price", "promo", "deal",
  ]);

  // Words that look ambiguous between English and Telex (repair only in VN context)
  const CONTEXTUAL_ENGLISH_WORDS = new Set([
    "hair", "bar", "car", "box", "max", "mix", "tax", "bus", "gas", "has", "his",
    "her", "sir", "see", "bee", "tee", "beef", "beer", "door", "soon", "moon", "noon",
    "boom", "loop", "boot", "root", "deep", "keep", "meet", "down", "town", "how",
    "now", "low", "cow", "bow", "row", "chef", "roof", "cats", "hats", "caps", "cups",
    "tops", "fans", "vans", "cans",
  ]);

  // Core Vietnamese structural & common place tokens (populated further on initConverterV3)
  const BASE_VN_TOKENS = new Set([
    "phuong", "xa", "thi", "tran", "dac", "khu", "quan", "huyen", "tinh", "thanh",
    "pho", "duong", "ngo", "ngach", "hem", "kiet", "thon", "ap", "xom", "ban",
    "khom", "to", "dan", "tang", "lau", "phong", "can", "ho", "chung", "cu",
    "toa", "nha", "so", "quoc", "lo", "dai", "viet", "nam", "ha", "noi", "chi",
    "minh", "sai", "gon", "da", "nang", "hai", "binh", "phuoc", "dong", "nai",
    "loc", "ninh", "ba", "dinh", "phuc", "hong", "xuan", "thuong", "nguyen",
    "trai", "tan", "lap", "cau", "giay", "hoan", "kiem", "tay", "long", "bien",
    "hoang", "mai", "bac", "tu", "liem", "go", "vap", "phu", "nhuan", "duong",
    "vung", "tau", "thua", "thien", "hue", "tho", "khanh", "hoa", "lam", "dak",
    "lak", "gia", "lai", "quang", "ngai", "thai", "tuyen", "lao", "cai", "yen",
    "bai", "son", "la", "dien", "lang", "cao", "bang", "kan", "vinh", "hung",
    "nghe", "an", "tri", "ninh", "thuan", "tien", "ben", "tre", "tra", "hau",
    "soc", "trang", "lieu", "ca", "mau", "kien", "thap", "chau", "quyen",
    "cong", "vi", "ngoc", "truc", "bach", "van", "mieu", "giang", "vo", "duc",
    "thanh", "o", "dien", "hoai", "bao", "my", "cuong", "tien", "phu", "tan",
  ]);

  // Real minority/ethnic VN place tokens containing oo/ee/w/r/j/s/z that MUST NOT be mutated
  const DEFAULT_EXACT_PROTECTED_ADMIN = new Set([
    "choong", "loong", "noong", "toong", "khoong", "nhoong", "dooc", "lee",
    "rooi", "cooih", "ee", "poo", "dee", "kroong", "hoong", "moong", "boong",
    "ngoong", "chais", "krong", "kar", "bar", "mar", "xier", "smar", "glar",
    "ver", "der", "ar", "tor", "pior", "uar", "piar", "huar", "wer", "cuor",
    "tar", "sar", "dar", "dur", "ktur", "sor", "braih", "jut", "jo", "jing",
    "wa", "wang", "wy", "ewi", "wil", "za", "zuoih",
  ]);

  const VALID_VN_NUCLEI = new Set([
    "a", "ai", "ao", "au", "ay",
    "e", "eo", "eu",
    "i", "ia", "ie", "ieu", "iu",
    "o", "oa", "oai", "oao", "oay", "oe", "oeo", "oi",
    "u", "ua", "uai", "uao", "uay", "ue", "ui", "uo", "uoi", "uou", "uu", "uy", "uya", "uye", "uyu",
    "y", "ya", "ye", "yeu",
  ]);

  const VALID_VN_ONSETS = new Set([
    "", "b", "c", "ch", "d", "g", "gh", "gi", "h", "k", "kh", "l", "m",
    "n", "ng", "ngh", "nh", "p", "ph", "q", "qu", "r", "s", "t", "th", "tr", "v", "x",
  ]);

  const VALID_VN_CODAS = new Set(["", "c", "ch", "k", "m", "n", "ng", "nh", "p", "t"]);

  const WARD_PREFIX_SET = new Set(["phuong", "xa", "thi tran", "dac khu", "p", "x", "tt", "dk", "ph"]);
  const DIST_PREFIX_SET = new Set(["quan", "huyen", "thi xa", "thanh pho", "q", "h", "tx", "tp"]);
  const PROV_PREFIX_SET = new Set(["tinh", "thanh pho", "tp", "t"]);
  const GRANULAR_SKIP_WORDS = new Set([
    "so", "nha", "duong", "pho", "ngo", "ngach", "hem", "kiet", "thon", "ap",
    "xom", "ban", "khom", "to", "tang", "lau", "phong", "can", "chung", "toa",
    "block", "lo", "khu", "kdt", "kcn", "kcx", "ql", "tl", "cho", "minh", "em",
    "chai", "goi", "hop", "cai", "chiec", "ship", "giao", "dat",
  ]);
  const AMBIGUOUS_SINGLE_WORD_ALIASES = new Set(["vinh"]);

  // URL / Email / Phone / House-slash / Numeric protection
  const PROTECTED_SPAN_RE =
    /(?:https?:[/]{2}[^\s,;]+|www\.[^\s,;]+|[^\s@]+@[^\s@]+\.[^\s@]{2,}|(?:\+?84|0)(?:[\s.\-]*\d){8,10}(?=\s|,|;|$)|[a-zA-Z0-9-]+\.(?:com|net|org|vn|edu\.vn|gov\.vn|io|co|info|biz|app|dev)(?:\/[^\s,;]*)?|\S*\/\S*|\b\d+[a-zA-Z0-9._-]*\b)/gi;

  const state = {
    initialized: false,
    c: null,
    adminTokens: new Set(BASE_VN_TOKENS),
    exactProtectedAdmin: new Set(DEFAULT_EXACT_PROTECTED_ADMIN),
    exactMap: new Map(), // norm -> { s, po, do, wo, pn, wn }
    delMap: new Map(),   // del1 -> string | string[]
    paMap: new Map(),    // alias -> canonical norm
    daMap: new Map(),    // alias -> canonical norm
    maxPhraseWords: 6,
    initMs: 0,
  };

  function stripDiacritics(str) {
    return (str || "")
      .replace(/đ/g, "d")
      .replace(/Đ/g, "D")
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "");
  }

  function normalizeText(str) {
    return stripDiacritics((str || "").trim())
      .toLowerCase()
      .replace(/\s+/g, " ");
  }

  function sameDigits(a, b) {
    const da = a.replace(/\D+/g, "");
    const db = b.replace(/\D+/g, "");
    return da === db;
  }

  /**
   * Checks whether `syl` (lowercase ASCII) is phonotactically a valid Vietnamese syllable.
   */
  function isValidVietnameseSyllable(syl) {
    if (!syl || syl.length > 8) return false;
    // Split onset + rime
    let onset = "";
    let i = 0;
    const two = syl.slice(0, 2);
    const three = syl.slice(0, 3);
    if (three === "ngh") {
      onset = "ngh";
      i = 3;
    } else if (two === "qu" || two === "gi") {
      // "gi" followed by consonant or end isn't valid, "gi" followed by vowel works as onset, or "g"+"i"
      if (two === "qu") {
        onset = "qu";
        i = 2;
      } else if (syl.length > 2 && /[aeiouy]/.test(syl[2])) {
        onset = "gi";
        i = 2;
      } else {
        onset = "g";
        i = 1;
      }
    } else if (
      two === "ch" || two === "gh" || two === "kh" || two === "ng" ||
      two === "nh" || two === "ph" || two === "th" || two === "tr"
    ) {
      onset = two;
      i = 2;
    } else if (!/[aeiouy]/.test(syl[0])) {
      onset = syl[0];
      i = 1;
    }

    if (!VALID_VN_ONSETS.has(onset)) return false;

    const rime = syl.slice(i);
    if (!rime) return false;

    const m = rime.match(/^([aeiouy]+)([a-z]*)$/);
    if (!m) return false;
    const nucleus = m[1];
    const coda = m[2];
    if (!VALID_VN_NUCLEI.has(nucleus)) return false;
    if (!VALID_VN_CODAS.has(coda)) return false;
    if (nucleus === "uu" && coda !== "") return false;
    return true;
  }

  function applyCasePattern(orig, repaired) {
    if (!orig || !repaired) return repaired;
    if (orig.length > 1 && orig === orig.toUpperCase() && /[A-Z]/.test(orig)) {
      return repaired.toUpperCase();
    }
    const first = orig[0];
    if (first >= "A" && first <= "Z") {
      return repaired[0].toUpperCase() + repaired.slice(1);
    }
    return repaired;
  }

  /**
   * Attempts Telex or VNI repair on a single non-protected alphabetic/VNI word token.
   * Returns `{ text, highConf }` or `null` if untouched.
   */
  function repairSingleToken(rawTok, hasVnSentenceContext) {
    if (!rawTok || rawTok.length < 2) return null;
    // Skip words already carrying Vietnamese Unicode accents
    if (/[^\x00-\x7f]/.test(rawTok)) return null;

    const low = rawTok.toLowerCase();

    // Protect known English words & minority place names
    if (ENGLISH_PROTECTED_WORDS.has(low)) {
      if (hasVnSentenceContext && CONTEXTUAL_ENGLISH_WORDS.has(low)) {
        // Allow contextual repair below only if it yields a known VN admin token
      } else {
        return null;
      }
    }
    if (state.exactProtectedAdmin.has(low)) return null;

    // 1. VNI check (digits embedded inside or at end of alphabetic syllable, e.g. lo6c5, phu7o7ng2, huye6n5, Tha5nh)
    if (/\d/.test(low)) {
      // Never touch short block/district codes like p12, q7, a5, b2, ct2, x1
      if (/^(?:[pqfkabcdthx]|ct|lk|bt|nv|ot)\d{1,3}[a-z]?$/i.test(low)) return null;
      const hasVniDiacritic = /d9|[aeiouy][6789]|[aeiouy][1-5](?:nh|ng|ch|[cmnpt])$/.test(low);
      const hasVniTone = /^[a-z]{2,}(?:[1-5]|[6789][a-z]*[1-5]?)$/.test(low) || /[aeiouy][1-9][a-z]*[1-5]?$/.test(low);
      if (!hasVniDiacritic && !(hasVniTone && hasVnSentenceContext)) return null;

      let v = low
        .replace(/d9/g, "d")
        .replace(/([aeiouy])[6789]/g, "$1")
        .replace(/([aeiouy])[1-5]([a-z]*)$/, "$1$2")
        .replace(/[1-5]+$/g, "");
      if (v !== low && !/\d/.test(v) && isValidVietnameseSyllable(v)) {
        if (hasVniDiacritic || state.adminTokens.has(v)) {
          return { text: applyCasePattern(rawTok, v), highConf: true };
        }
      }
      return null;
    }

    // Pure alphabetic ASCII only from here on
    if (!/^[a-z]+$/.test(low)) return null;

    // Fast skip if no Telex signature at all
    const hasDoubleKey = /(?:^dd|aa|ee|oo|aw|ow|uw|uow|uwow)/.test(low);
    const hasTrailingW = /[auo][a-z]*w[fjrsxz]?$/.test(low);
    const hasToneMarker = /^[a-z]*[aeiouy][a-z]*[fjrsxz]/.test(low);
    if (!hasDoubleKey && !hasTrailingW && !hasToneMarker) return null;

    // If token is already a valid admin token AND has no double-vowel/dd/tone-key typo, leave untouched
    if (state.adminTokens.has(low) && !hasDoubleKey && !hasTrailingW && !/[fjz]/.test(low)) {
      return null;
    }

    // Parse onset so we never strip initial r/s/x or tr/kr/gr/pr/br/dr/hr/sr 'r'
    let onsetEnd = 0;
    if (low.startsWith("dd") && low.length >= 3 && /[aeiouy]/.test(low[2])) {
      onsetEnd = 2; // handled via dd->d below
    } else if (/^(?:ngh|ch|gh|kh|ng|nh|ph|th|tr|qu|gi)/.test(low)) {
      onsetEnd = low.startsWith("ngh") ? 3 : 2;
    } else if (!/[aeiouy]/.test(low[0])) {
      onsetEnd = 1;
    }

    let onset = low.slice(0, onsetEnd);
    let rime = low.slice(onsetEnd);

    if (onset === "dd") onset = "d";
    else if (onset.length >= 2 && onset[0] === onset[1] && onset[0] !== "d") {
      // Stuck consonant onset key? Rarely, leave alone unless needed
    }

    // Collapse stuck duplicate tone/coda chars first cleanly:
    // Strip any Telex tone keys (f, j, r, s, x, z) anywhere in the rime
    const rimeNoTone = rime.replace(/[fjrsxz]+/g, "");
    const strippedTone = rimeNoTone !== rime;
    const hadStrongTone = /[fjz]/.test(rime);

    // Handle Telex horn/breve/circumflex sequences in rimeNoTone
    let r = rimeNoTone
      .replace(/uwow/g, "uo")
      .replace(/uow/g, "uo")
      .replace(/uw/g, "u")
      .replace(/aw/g, "a")
      .replace(/ow/g, "o")
      .replace(/aa+/g, "a")
      .replace(/ee+/g, "e")
      .replace(/oo+/g, "o");

    // Handle delayed 'w' typed after coda (e.g. "dducws" -> "ducw" -> "duc", "phuongw" -> "phuong")
    if (/[auo][a-z]*w$/.test(r)) {
      r = r.slice(0, -1);
    } else if (/[auo]w[a-z]+$/.test(r)) {
      r = r.replace(/w/g, "");
    }

    // Collapse stuck double trailing consonants (e.g. "binhh" -> "binh", "locc" -> "loc", "huyenn" -> "huyen")
    r = r.replace(/([a-z])\1+$/g, "$1");

    const cand = onset + r;
    if (cand === low) return null;
    if (!isValidVietnameseSyllable(cand)) return null;

    // Guard contextual English words (like "hair" -> "hai", "box" -> "bo")
    if (CONTEXTUAL_ENGLISH_WORDS.has(low)) {
      if (hasVnSentenceContext && state.adminTokens.has(cand)) {
        return { text: applyCasePattern(rawTok, cand), highConf: false };
      }
      return null;
    }

    // If the ONLY transformation was stripping trailing single 's' or 'r' (no dd/aa/ee/oo/aw/ow/uw/f/j/x/z),
    // avoid turning arbitrary English plurals (e.g. "cats", "tops", "cars") into VN words unless VN context or strong VN stem!
    const onlyTrailingSR =
      !hasDoubleKey &&
      !hasTrailingW &&
      !hadStrongTone &&
      !low.startsWith("dd") &&
      (low === cand + "s" || low === cand + "r");

    if (onlyTrailingSR) {
      const strongVn = state.adminTokens.has(cand) && cand.length >= 3;
      if (!strongVn && !(hasVnSentenceContext && state.adminTokens.has(cand))) {
        return null;
      }
      // Even if strongVn, guard common 3-4 letter English stems + s/r (e.g. "bars", "cars", "hats", "cats", "vans", "fans")
      if (ENGLISH_PROTECTED_WORDS.has(cand) || ENGLISH_PROTECTED_WORDS.has(low)) {
        if (!hasVnSentenceContext) return null;
      }
    }

    const highConf = hasDoubleKey || hasTrailingW || hadStrongTone || low.startsWith("dd") || state.adminTokens.has(cand);
    return { text: applyCasePattern(rawTok, cand), highConf };
  }

  /**
   * Repairs Vietnamese Telex/VNI stuck-key typos without mutating URLs, emails,
   * house numbers/slashes (e.g. 45/2B), phones, or common English words.
   */
  function repairTelexTypos(text) {
    if (!text || typeof text !== "string") return "";

    // Fast path: pure ASCII with no spaces/letters? Or quickly split protected spans vs words
    const hasVnAccents = /[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]/i.test(text);

    // First collect protected spans (URLs, emails, phones, slash tokens, digit-leading tokens)
    const segments = [];
    let lastIdx = 0;
    PROTECTED_SPAN_RE.lastIndex = 0;
    let m;
    while ((m = PROTECTED_SPAN_RE.exec(text)) !== null) {
      if (m.index > lastIdx) {
        segments.push({ protected: false, text: text.slice(lastIdx, m.index) });
      }
      segments.push({ protected: true, text: m[0] });
      lastIdx = m.index + m[0].length;
    }
    if (lastIdx < text.length) {
      segments.push({ protected: false, text: text.slice(lastIdx) });
    }

    // Check if sentence already carries VN/Telex cues so contextual words ("hair phongf") can safely repair
    let hasVnContext = hasVnAccents;
    if (!hasVnContext) {
      for (const seg of segments) {
        if (seg.protected) continue;
        const words = seg.text.match(/[a-zA-Z0-9]+/g);
        if (!words) continue;
        for (const w of words) {
          const wl = w.toLowerCase();
          if (
            state.adminTokens.has(wl) ||
            /(?:^dd|uow|uwow|phuow|dduow|loocj|huyeen|quaan|thanhf|binhf|nooij|haf)/.test(wl) ||
            /[a-z]+[fj]$/.test(wl) ||
            /d9|[aeiouy][6789]|[aeiouy][1-5](?:nh|ng|ch|[cmnpt])$/.test(wl)
          ) {
            hasVnContext = true;
            break;
          }
        }
        if (hasVnContext) break;
      }
    }

    let out = "";
    for (const seg of segments) {
      if (seg.protected) {
        out += seg.text;
        continue;
      }
      out += seg.text.replace(/[A-Za-z][A-Za-z0-9]*/g, (tok) => {
        const res = repairSingleToken(tok, hasVnContext);
        return res ? res.text : tok;
      });
    }
    return out;
  }

  /**
   * Initializes compact 2025 converter indexes & SymSpell-style 1-edit delete dictionary (< 25 ms).
   */
  function initConverterV3(converterJson) {
    if (!converterJson || !Array.isArray(converterJson.wo) || !Array.isArray(converterJson.eo)) {
      throw new Error("Invalid vn_admin_v3_converter.json payload");
    }
    const t0 = typeof performance !== "undefined" ? performance.now() : Date.now();
    const c = converterJson;
    state.c = c;

    const exactMap = new Map();
    const tokenDelMap = new Map();
    const adminTokens = new Set(BASE_VN_TOKENS);
    const exactProtectedAdmin = new Set(DEFAULT_EXACT_PROTECTED_ADMIN);
    let maxWords = 1;

    const addGroup = (arr, kind) => {
      for (let i = 0; i < arr.length; i++) {
        const s = arr[i];
        if (!s) continue;
        let rec = exactMap.get(s);
        if (!rec) {
          rec = { s, po: -1, do: -1, wo: -1, pn: -1, wn: -1 };
          exactMap.set(s, rec);
        }
        if (rec[kind] === -1) rec[kind] = i;
      }
    };

    addGroup(c.po, "po");
    addGroup(c.do, "do");
    addGroup(c.wo, "wo");
    addGroup(c.pn, "pn");
    addGroup(c.wn, "wn");

    const placeTokens = new Set();
    for (const s of exactMap.keys()) {
      let start = 0;
      let wc = 1;
      const len = s.length;
      for (let k = 0; k <= len; k++) {
        if (k === len || s.charCodeAt(k) === 32) {
          if (k > start) {
            const tok = s.slice(start, k);
            placeTokens.add(tok);
            adminTokens.add(tok);
          }
          if (k < len) wc++;
          start = k + 1;
        }
      }
      if (wc > maxWords) maxWords = wc;
    }

    // Register hyphen-collapsed forms for exact matching (e.g. "ba ria vung tau")
    for (const [key, rec] of exactMap) {
      if (key.indexOf("-") !== -1) {
        const noH = key.replace(/\s*-\s*/g, " ").replace(/\s+/g, " ").trim();
        if (noH && noH !== key && !exactMap.has(noH)) {
          exactMap.set(noH, { s: noH, po: rec.po, do: rec.do, wo: rec.wo, pn: rec.pn, wn: rec.wn });
        }
      }
    }

    // Aliases
    const paMap = new Map();
    if (c.pa) {
      for (const [alias, canon] of Object.entries(c.pa)) {
        paMap.set(
          alias.replace(/\s*-\s*/g, " ").replace(/\s+/g, " ").trim(),
          canon.replace(/\s*-\s*/g, " ").replace(/\s+/g, " ").trim()
        );
      }
    }
    const daMap = new Map();
    if (c.da) {
      for (const [alias, canon] of Object.entries(c.da)) {
        daMap.set(alias.replace(/\s+/g, " ").trim(), canon.replace(/\s+/g, " ").trim());
      }
    }

    // Build compact 1-delete dictionary over unique place tokens (~1,073 tokens -> ~1,879 keys, < 7 ms!)
    for (const tok of placeTokens) {
      const len = tok.length;
      if (len < 2 || /^\d+$/.test(tok)) continue;
      for (let i = 0; i < len; i++) {
        const ch = tok.charCodeAt(i);
        if (ch >= 48 && ch <= 57) continue; // never delete digits
        const d = tok.slice(0, i) + tok.slice(i + 1);
        const cur = tokenDelMap.get(d);
        if (cur === undefined) {
          tokenDelMap.set(d, tok);
        } else if (typeof cur === "string") {
          if (cur !== tok) tokenDelMap.set(d, [cur, tok]);
        } else if (cur[cur.length - 1] !== tok) {
          cur.push(tok);
        }
      }
    }

    state.exactMap = exactMap;
    state.delMap = tokenDelMap;
    state.placeTokens = placeTokens;
    state.paMap = paMap;
    state.daMap = daMap;
    state.adminTokens = adminTokens;
    state.exactProtectedAdmin = exactProtectedAdmin;
    state.maxPhraseWords = Math.min(7, maxWords);
    state.initialized = true;
    state.initMs = +(((typeof performance !== "undefined" ? performance.now() : Date.now()) - t0)).toFixed(2);
    return true;
  }

  function ensureConverterLoaded() {
    if (state.initialized) return true;
    if (typeof require !== "undefined" && typeof __dirname !== "undefined") {
      try {
        const fs = require("fs");
        const path = require("path");
        const candidates = [
          path.join(__dirname, "..", "dist", "model", "vn_admin_v3_converter.json"),
          path.join(__dirname, "model", "vn_admin_v3_converter.json"),
        ];
        for (const p of candidates) {
          if (fs.existsSync(p)) {
            const json = JSON.parse(fs.readFileSync(p, "utf8"));
            return initConverterV3(json);
          }
        }
      } catch (_) {
        // Fall through cleanly
      }
    }
    return false;
  }

  function isConverterInitialized() {
    return state.initialized;
  }

  /**
   * True if Damerau-Levenshtein distance(a, b) <= 1 (allows 1 insertion, deletion,
   * substitution, or adjacent transposition, and enforces matching digits).
   */
  function isDamerauLevenshtein1(a, b) {
    if (a === b) return true;
    const la = a.length;
    const lb = b.length;
    if (Math.abs(la - lb) > 1) return false;
    if (!sameDigits(a, b)) return false;

    if (la === lb) {
      let diff1 = -1;
      let diff2 = -1;
      for (let i = 0; i < la; i++) {
        if (a.charCodeAt(i) !== b.charCodeAt(i)) {
          if (diff1 === -1) diff1 = i;
          else if (diff2 === -1) diff2 = i;
          else return false;
        }
      }
      if (diff2 === -1) return true; // 1 substitution
      // Adjacent transposition check
      return diff2 === diff1 + 1 && a[diff1] === b[diff2] && a[diff2] === b[diff1];
    }

    // |la - lb| === 1
    const shorter = la < lb ? a : b;
    const longer = la < lb ? b : a;
    let i = 0;
    let j = 0;
    let skipped = false;
    while (i < shorter.length && j < longer.length) {
      if (shorter.charCodeAt(i) === longer.charCodeAt(j)) {
        i++;
        j++;
      } else if (!skipped) {
        skipped = true;
        j++;
      } else {
        return false;
      }
    }
    return true;
  }

  function tokenizePhraseStream(norm) {
    const s = norm
      .replace(/[,;|]+/g, " , ")
      .replace(/\s+-\s+/g, " ")
      .replace(/(^|\s)tp\.?\s*hcm(?=\s|$)/g, "$1tphcm")
      .replace(/(^|\s)tp\.?\s*hn(?=\s|$)/g, "$1tphn")
      .replace(/(^|\s)br\s*-\s*vt(?=\s|$)/g, "$1brvt")
      .replace(/(^|\s)(x|p|q|h|t|tt|tx|tp|dk)\.(?=[a-z0-9])/g, "$1$2. ")
      .replace(/(^|\s)(p|q)(\d{1,2})(?=\s|$)/g, "$1$2 $3")
      .replace(/\.(?=\s|$)/g, "")
      .replace(/\s+/g, " ")
      .trim();
    return s ? s.split(" ") : [];
  }

  function detectPrefixBefore(tokens, idx) {
    if (idx <= 0) return { kind: null, rawPrefix: null, start: idx };
    const t1 = tokens[idx - 1];
    if (t1 === ",") return { kind: null, rawPrefix: null, start: idx };
    if (idx >= 2 && tokens[idx - 2] !== ",") {
      const t2 = tokens[idx - 2] + " " + t1;
      if (t2 === "thi tran" || t2 === "dac khu") {
        return { kind: "ward", rawPrefix: t2, start: idx - 2 };
      }
      if (t2 === "thi xa" || t2 === "thanh pho") {
        return { kind: t2 === "thanh pho" ? "dist_or_prov" : "dist", rawPrefix: t2, start: idx - 2 };
      }
    }
    if (WARD_PREFIX_SET.has(t1)) return { kind: "ward", rawPrefix: t1, start: idx - 1 };
    if (DIST_PREFIX_SET.has(t1)) return { kind: t1 === "tp" ? "dist_or_prov" : "dist", rawPrefix: t1, start: idx - 1 };
    if (PROV_PREFIX_SET.has(t1)) return { kind: "prov", rawPrefix: t1, start: idx - 1 };
    return { kind: null, rawPrefix: null, start: idx };
  }

  function lookupExactRec(phrase) {
    let rec = state.exactMap.get(phrase);
    if (rec) return { rec, canonical: phrase };
    const pa = state.paMap.get(phrase);
    if (pa) {
      rec = state.exactMap.get(pa);
      if (rec) return { rec, canonical: pa };
    }
    const da = state.daMap.get(phrase);
    if (da) {
      rec = state.exactMap.get(da);
      if (rec) return { rec, canonical: da };
    }
    return null;
  }

  function collect1EditRecords(phrase) {
    if (phrase.length < 4) return [];
    if (/^\d+$/.test(phrase)) return [];
    const words = phrase.split(" ");
    if (words.some((w) => GRANULAR_SKIP_WORDS.has(w))) return [];

    const candidates = new Set();
    const tokCands = new Set();
    const addTokenHits = (val) => {
      if (typeof val === "string") tokCands.add(val);
      else if (Array.isArray(val)) {
        for (let i = 0; i < val.length; i++) tokCands.add(val[i]);
      }
    };

    // 1. Single-token 1-edit lookup via compact tokenDelMap
    for (let wi = 0; wi < words.length; wi++) {
      const w = words[wi];
      if (!w || /^\d+$/.test(w)) continue;
      tokCands.clear();

      addTokenHits(state.delMap.get(w));
      for (let k = 0; k < w.length; k++) {
        const ch = w.charCodeAt(k);
        if (ch >= 48 && ch <= 57) continue;
        const d = w.slice(0, k) + w.slice(k + 1);
        if (state.placeTokens && state.placeTokens.has(d)) tokCands.add(d);
        addTokenHits(state.delMap.get(d));
      }

      for (const rep of tokCands) {
        if (rep === w) continue;
        const orig = words[wi];
        words[wi] = rep;
        const candPhrase = words.join(" ");
        words[wi] = orig;
        if (candPhrase.length >= 4 && state.exactMap.has(candPhrase)) {
          candidates.add(candPhrase);
        }
      }
    }

    // 2. Missing space inside single token (e.g. "locthanh" -> "loc thanh")
    if (words.length === 1 && phrase.length >= 5) {
      for (let k = 2; k <= phrase.length - 2; k++) {
        const sp = phrase.slice(0, k) + " " + phrase.slice(k);
        if (state.exactMap.has(sp)) candidates.add(sp);
      }
    }

    // 3. Accidental extra space between adjacent tokens (e.g. "loc th anh" -> "loc thanh")
    if (words.length >= 2) {
      for (let wi = 0; wi < words.length - 1; wi++) {
        const joined = words.slice(0, wi).concat([words[wi] + words[wi + 1]], words.slice(wi + 2)).join(" ");
        if (joined.length >= 4 && state.exactMap.has(joined)) {
          candidates.add(joined);
        }
      }
    }

    const out = [];
    for (const cand of candidates) {
      if (cand === phrase) continue;
      if (!isDamerauLevenshtein1(phrase, cand)) continue;
      const rec = state.exactMap.get(cand);
      if (rec) out.push({ rec, canonical: cand, editCost: 1 });
    }
    return out;
  }

  function scanSpans(tokens, allowFuzzy) {
    const wardsOld = [];
    const wardsNew = [];
    const districts = [];
    const provincesOld = [];
    const provincesNew = [];
    const n = tokens.length;

    const pushRec = (rec, canonical, editCost, i, len, pfx) => {
      const phraseWords = canonical.split(" ").length;
      const hasNonWardPrefix = pfx.kind === "dist" || pfx.kind === "prov" || pfx.kind === "dist_or_prov";
      const hasWardPrefix = pfx.kind === "ward";
      const hasDistPrefix = pfx.kind === "dist" || pfx.kind === "dist_or_prov";
      const hasProvPrefix = pfx.kind === "prov" || pfx.kind === "dist_or_prov";

      if (rec.wo !== -1 && !hasNonWardPrefix && !/^\d+$/.test(canonical)) {
        const spanStart = hasWardPrefix ? pfx.start : i;
        wardsOld.push({
          id: rec.wo,
          name: canonical,
          start: spanStart,
          end: i + len,
          wordStart: i,
          wordCount: phraseWords,
          prefixed: hasWardPrefix,
          rawPrefix: pfx.rawPrefix,
          editCost,
        });
      }
      if (rec.wn !== -1 && !hasNonWardPrefix && !/^\d+$/.test(canonical)) {
        const spanStart = hasWardPrefix ? pfx.start : i;
        wardsNew.push({
          id: rec.wn,
          name: canonical,
          start: spanStart,
          end: i + len,
          wordStart: i,
          wordCount: phraseWords,
          prefixed: hasWardPrefix,
          rawPrefix: pfx.rawPrefix,
          editCost,
        });
      }
      if (rec.do !== -1) {
        const isAmbig = phraseWords === 1 && AMBIGUOUS_SINGLE_WORD_ALIASES.has(canonical) && !hasDistPrefix;
        const did = rec.do;
        if (hasDistPrefix) {
          districts.push({
            id: did,
            name: canonical,
            start: pfx.start,
            end: i + len,
            wordStart: i,
            wordCount: phraseWords,
            prefixed: true,
            rawPrefix: pfx.rawPrefix,
            ambiguousSingle: false,
            editCost,
          });
        }
        districts.push({
          id: did,
          name: canonical,
          start: i,
          end: i + len,
          wordStart: i,
          wordCount: phraseWords,
          prefixed: false,
          rawPrefix: null,
          ambiguousSingle: isAmbig,
          editCost,
        });
      }
      if (rec.po !== -1) {
        const isAmbig = phraseWords === 1 && AMBIGUOUS_SINGLE_WORD_ALIASES.has(canonical) && !hasProvPrefix;
        const pid = rec.po;
        if (hasProvPrefix) {
          provincesOld.push({
            id: pid,
            name: canonical,
            start: pfx.start,
            end: i + len,
            wordStart: i,
            wordCount: phraseWords,
            prefixed: true,
            ambiguousSingle: false,
            editCost,
          });
        }
        provincesOld.push({
          id: pid,
          name: canonical,
          start: i,
          end: i + len,
          wordStart: i,
          wordCount: phraseWords,
          prefixed: false,
          ambiguousSingle: isAmbig,
          editCost,
        });
      }
      if (rec.pn !== -1) {
        const isAmbig = phraseWords === 1 && AMBIGUOUS_SINGLE_WORD_ALIASES.has(canonical) && !hasProvPrefix;
        const pidN = rec.pn;
        if (hasProvPrefix) {
          provincesNew.push({
            id: pidN,
            name: canonical,
            start: pfx.start,
            end: i + len,
            wordStart: i,
            wordCount: phraseWords,
            prefixed: true,
            ambiguousSingle: false,
            editCost,
          });
        }
        provincesNew.push({
          id: pidN,
          name: canonical,
          start: i,
          end: i + len,
          wordStart: i,
          wordCount: phraseWords,
          prefixed: false,
          ambiguousSingle: isAmbig,
          editCost,
        });
      }
    };

    for (let i = 0; i < n; i++) {
      if (tokens[i] === ",") continue;
      const pfx = detectPrefixBefore(tokens, i);
      const maxLen = Math.min(state.maxPhraseWords, n - i);

      for (let len = maxLen; len >= 1; len--) {
        const slice = tokens.slice(i, i + len);
        if (slice.includes(",")) continue;
        const phrase = slice.join(" ");

        // Exact match (plus independent da/pa alias lookup in case phrase also matches a ward name, e.g. 'phu quy' -> 'phu qui')
        const exact = lookupExactRec(phrase);
        if (exact) {
          pushRec(exact.rec, exact.canonical, 0, i, len, pfx);
        }
        const daCanon = state.daMap.get(phrase);
        if (daCanon && (!exact || daCanon !== exact.canonical)) {
          const daRec = state.exactMap.get(daCanon);
          if (daRec && daRec.do !== -1) {
            pushRec({ s: daRec.s, po: -1, do: daRec.do, wo: -1, pn: -1, wn: -1 }, daCanon, 0, i, len, pfx);
          }
        }
        const paCanon = state.paMap.get(phrase);
        if (paCanon && (!exact || paCanon !== exact.canonical)) {
          const paRec = state.exactMap.get(paCanon);
          if (paRec && (paRec.po !== -1 || paRec.pn !== -1)) {
            pushRec({ s: paRec.s, po: paRec.po, do: -1, wo: -1, pn: paRec.pn, wn: -1 }, paCanon, 0, i, len, pfx);
          }
        }

        // Numeric prefixed ward ("p 12" -> "phuong 12") or district ("q 7" -> "quan 7")
        if (len === 1 && /^\d{1,2}$/.test(phrase)) {
          if (pfx.kind === "ward") {
            const numW = lookupExactRec("phuong " + phrase);
            if (numW) pushRec(numW.rec, numW.canonical, 0, i, 1, pfx);
          }
          if (pfx.kind === "dist" || tokens[i - 1] === "q" || tokens[i - 1] === "quan") {
            const numD = lookupExactRec("quan " + phrase);
            if (numD) pushRec(numD.rec, numD.canonical, 0, i, 1, pfx);
          }
        }

        // 1-edit fuzzy recovery
        if (allowFuzzy && !exact && phrase.length >= 4) {
          // Don't fuzzy-match bare admin prefixes or phrases starting with an explicit admin prefix token ("xa tan", "q 1")
          const firstTok = slice[0];
          const startsWithAdminPrefix =
            WARD_PREFIX_SET.has(firstTok) ||
            DIST_PREFIX_SET.has(firstTok) ||
            PROV_PREFIX_SET.has(firstTok);
          if (!startsWithAdminPrefix && !WARD_PREFIX_SET.has(phrase) && !DIST_PREFIX_SET.has(phrase) && !PROV_PREFIX_SET.has(phrase)) {
            const fuzHits = collect1EditRecords(phrase);
            for (const h of fuzHits) {
              pushRec(h.rec, h.canonical, h.editCost, i, len, pfx);
            }
          }
        }
      }
    }

    return { wardsOld, wardsNew, districts, provincesOld, provincesNew };
  }

  function disjoint(a, b) {
    return a.end <= b.start || b.end <= a.start;
  }

  function formatWard2025(widN, pidN) {
    const c = state.c;
    const entries = c.en[widN] || [];
    const match = entries.find((x) => x[0] === pidN) || entries[0];
    const codeDisp = match && c.wn_code_disp ? c.wn_code_disp[String(match[4])] : null;
    const bare = codeDisp || c.wn_disp[widN];
    if (/^(?:Phường|Xã|Thị trấn|Đặc khu)\s+/i.test(bare)) return bare;
    const wt = match ? c.wt[match[5]] : "Xã";
    return `${wt} ${bare}`;
  }

  function formatOldWard(widO, widN, pidN, rawPrefix) {
    const c = state.c;
    let bare = c.wo_disp[widO];
    // Fix tone collision when wo_disp collapsed two tone variants (e.g. Lộc Thanh vs Lộc Thạnh)
    if (widN !== undefined && widN !== null) {
      const wnBare = c.wn_disp[widN];
      if (wnBare && normalizeText(wnBare) === c.wo[widO]) {
        bare = wnBare;
      }
    }
    if (/^(?:Phường|Xã|Thị trấn|Đặc khu)\s+/i.test(bare)) return bare;
    let wt = "Xã";
    if (rawPrefix === "p" || rawPrefix === "ph" || rawPrefix === "phuong") wt = "Phường";
    else if (rawPrefix === "x" || rawPrefix === "xa") wt = "Xã";
    else if (rawPrefix === "tt" || rawPrefix === "thi tran") wt = "Thị trấn";
    else if (rawPrefix === "dk" || rawPrefix === "dac khu") wt = "Đặc khu";
    else if (widN !== undefined && widN !== null) {
      const entries = c.en[widN] || [];
      const match = entries.find((x) => x[0] === pidN) || entries[0];
      if (match) wt = c.wt[match[5]];
    }
    return `${wt} ${bare}`;
  }

  function formatOldDistrict(didO, rawPrefix) {
    const c = state.c;
    const bare = c.do_disp[didO];
    if (/^(?:Quận|Huyện|Thị xã|Thành phố)\s+/i.test(bare)) return bare;
    if (rawPrefix === "q" || rawPrefix === "quan") return `Quận ${bare}`;
    if (rawPrefix === "h" || rawPrefix === "huyen") return `Huyện ${bare}`;
    if (rawPrefix === "tx" || rawPrefix === "thi xa") return `Thị xã ${bare}`;
    if (rawPrefix === "tp" || rawPrefix === "thanh pho") return `Thành phố ${bare}`;
    const normD = c.do[didO];
    if (LEGACY_QUAN_NAMES.has(normD)) return `Quận ${bare}`;
    if (LEGACY_TP_NAMES.has(normD)) return `Thành phố ${bare}`;
    return `Huyện ${bare}`;
  }

  function formatProvinceWithLevel(pnDisp, pnNorm) {
    if (CENTRALLY_GOVERNED_2025.has(pnNorm)) {
      return pnDisp === "Hồ Chí Minh" ? "TP. Hồ Chí Minh" : pnDisp;
    }
    return `Tỉnh ${pnDisp}`;
  }

  function shortDistrictLabel(didO, rawPrefix) {
    const full = formatOldDistrict(didO, rawPrefix);
    return full
      .replace(/^Huyện /, "H. ")
      .replace(/^Quận /, "Q. ")
      .replace(/^Thị xã /, "TX. ")
      .replace(/^Thành phố /, "TP. ");
  }

  function buildConversionFromEdge(widO, didO, pidO, streetPrefix, rawWardPrefix, rawDistPrefix, rawLower) {
    const c = state.c;
    const edges = (c.eo[widO] || []).filter((e) => e[0] === didO && e[1] === pidO);
    if (!edges.length) return null;

    // Sort with is_default (e[5] === 1) first
    const sorted = edges.slice().sort((a, b) => b[5] - a[5]);
    const primary = sorted[0];
    const widN = primary[2];
    const pidN = primary[3];
    const isDivided = primary[4] === 1 || sorted.length > 1;

    const oldWard = formatOldWard(widO, widN, pidN, rawWardPrefix, rawLower);
    const oldDist = formatOldDistrict(didO, rawDistPrefix);
    const oldProv = c.po_disp[pidO];
    const oldFormatted = `${oldWard}, ${oldDist}, ${oldProv}`;

    const newWard = formatWard2025(widN, pidN);
    const newProv = c.pn_disp[pidN];
    const newAdminFormatted = `${newWard}, ${newProv}`;
    const newFormatted = streetPrefix ? `${streetPrefix}, ${newAdminFormatted}` : newAdminFormatted;

    const candidates = sorted.map((e) => {
      const cw = formatWard2025(e[2], e[3]);
      const cp = c.pn_disp[e[3]];
      const enEntry = (c.en[e[2]] || []).find((x) => x[0] === e[3]) || (c.en[e[2]] || [])[0];
      return {
        new_ward: cw,
        new_province: cp,
        new_formatted: streetPrefix ? `${streetPrefix}, ${cw}, ${cp}` : `${cw}, ${cp}`,
        new_admin_formatted: `${cw}, ${cp}`,
        is_default: e[5] === 1 || sorted.length === 1,
        lat: enEntry ? enEntry[1] : null,
        lon: enEntry ? enEntry[2] : null,
        area_km2: enEntry ? enEntry[3] : null,
        ward_code: enEntry ? enEntry[4] : null,
      };
    });

    return {
      converted: true,
      already_2025: false,
      old_formatted: oldFormatted,
      new_formatted: newFormatted,
      new_admin_formatted: newAdminFormatted,
      new_ward: newWard,
      new_province: newProv,
      old_ward: oldWard,
      old_district: oldDist,
      old_province: oldProv,
      is_divided: isDivided,
      candidates,
    };
  }

  function resolveBestHierarchy(spans) {
    const c = state.c;
    const { wardsOld, wardsNew, districts, provincesOld, provincesNew } = spans;

    // 1. Legacy Ward + District (+ optional Province)
    let bestWD = null;
    for (const w of wardsOld) {
      const edges = c.eo[w.id];
      if (!edges || !edges.length) continue;
      for (const d of districts) {
        if (!disjoint(w, d) || w.start > d.start) continue;
        const matchingEdges = edges.filter((e) => e[0] === d.id);
        if (!matchingEdges.length) continue;

        for (const e of matchingEdges) {
          const pidO = e[1];
          let provSpan = null;
          for (const p of provincesOld) {
            if (p.id === pidO && disjoint(w, p) && disjoint(d, p) && d.start <= p.start) {
              if (!provSpan || p.editCost < provSpan.editCost) provSpan = p;
            }
          }
          const totalEdits = w.editCost + d.editCost + (provSpan ? provSpan.editCost : 0);
          if (totalEdits > 2) continue;
          const score =
            (w.end - w.start) * 4 +
            (d.end - d.start) * 4 +
            (provSpan ? (provSpan.end - provSpan.start) * 4 + 6 : 0) +
            (w.prefixed ? 3 : 0) +
            (d.prefixed ? 3 : 0) -
            totalEdits * 12;
          if (!bestWD || score > bestWD.score) {
            bestWD = {
              type: provSpan ? "ward_district_province" : "ward_district",
              ward: w,
              district: d,
              province: provSpan,
              widO: w.id,
              didO: d.id,
              pidO,
              totalEdits,
              score,
            };
          }
        }
      }
    }
    if (bestWD) return bestWD;

    // 2. District + Province (checked before unprefixed Ward + Province so "Lộc Ninh, Bình Phước" is District+Province)
    let bestDP = null;
    for (const d of districts) {
      const dpPids = c.dp[d.id];
      if (!dpPids || !dpPids.length) continue;
      for (const p of provincesOld) {
        if (!disjoint(d, p) || d.start > p.start) continue;
        if (!dpPids.includes(p.id)) continue;
        const totalEdits = d.editCost + p.editCost;
        if (totalEdits > 1) continue;
        const score = (d.end - d.start) * 4 + (p.end - p.start) * 4 + (d.prefixed ? 4 : 0) - totalEdits * 12;
        if (!bestDP || score > bestDP.score) {
          bestDP = {
            type: "district_province",
            district: d,
            province: p,
            didO: d.id,
            pidO: p.id,
            totalEdits,
            score,
          };
        }
      }
    }

    // 3. 2025 2-tier Ward + Province (when explicit or non-shadowed by district+province)
    let bestWP2025 = null;
    for (const wn of wardsNew) {
      if (!wn.prefixed && wn.wordCount < 2) continue;
      const entries = c.en[wn.id];
      if (!entries || !entries.length) continue;
      for (const pn of provincesNew) {
        if (!disjoint(wn, pn) || wn.start > pn.start) continue;
        if (!entries.some((x) => x[0] === pn.id)) continue;
        const shadowed =
          !wn.prefixed &&
          bestDP &&
          !(wn.end <= bestDP.district.start || bestDP.district.end <= wn.start);
        if (shadowed) continue;
        const totalEdits = wn.editCost + pn.editCost;
        if (totalEdits > 1) continue;
        const score =
          (wn.end - wn.start) * 4 +
          (pn.end - pn.start) * 4 +
          (wn.prefixed ? 4 : 0) +
          (pn.prefixed ? 2 : 0) -
          totalEdits * 12;
        if (!bestWP2025 || score > bestWP2025.score) {
          bestWP2025 = {
            type: "ward_province_2025",
            ward: wn,
            province: pn,
            widN: wn.id,
            pidN: pn.id,
            totalEdits,
            score,
          };
        }
      }
    }

    // 4. Legacy Ward + Province (missing district)
    let bestWPOld = null;
    for (const w of wardsOld) {
      if (!w.prefixed && w.wordCount < 2) continue;
      const edges = c.eo[w.id];
      if (!edges || !edges.length) continue;
      for (const p of provincesOld) {
        if (!disjoint(w, p) || w.start > p.start) continue;
        const e = edges.find((x) => x[1] === p.id);
        if (!e) continue;
        const shadowed =
          !w.prefixed &&
          bestDP &&
          !(w.end <= bestDP.district.start || bestDP.district.end <= w.start);
        if (shadowed) continue;
        const totalEdits = w.editCost + p.editCost;
        if (totalEdits > 1) continue;
        const score =
          (w.end - w.start) * 4 +
          (p.end - p.start) * 4 +
          (w.prefixed ? 4 : 0) +
          (p.prefixed ? 2 : 0) -
          totalEdits * 12;
        if (!bestWPOld || score > bestWPOld.score) {
          bestWPOld = {
            type: "ward_province",
            ward: w,
            province: p,
            widO: w.id,
            didO: e[0],
            pidO: p.id,
            totalEdits,
            score,
          };
        }
      }
    }

    if (bestWPOld && (!bestWP2025 || bestWPOld.score > bestWP2025.score)) {
      return bestWPOld;
    }
    if (bestWP2025) return bestWP2025;

    // 5. Island special zone check (if District+Province is one of the 5 island special zones)
    if (bestDP) {
      const isleHit = (c.isle || []).find((x) => x[0] === bestDP.didO && x[1] === bestDP.pidO);
      if (isleHit) {
        return {
          type: "island_special",
          district: bestDP.district,
          province: bestDP.province,
          didO: isleHit[0],
          pidO: isleHit[1],
          widN: isleHit[2],
          pidN: isleHit[3],
          totalEdits: bestDP.totalEdits,
          score: bestDP.score + 5,
        };
      }
      return bestDP;
    }

    // 5b. Unprovinced explicit 'Đặc khu <name>' -> resolve unique 2025 wt === 3 ("Đặc khu") entry
    let bestUnprovDacKhu = null;
    for (const wn of wardsNew) {
      if (!wn.prefixed || (wn.rawPrefix !== "dac khu" && wn.rawPrefix !== "dk")) continue;
      const entries = c.en[wn.id] || [];
      const dkRows = entries.filter((x) => x[5] === 3);
      if (dkRows.length !== 1) continue;
      const score = (wn.end - wn.start) * 4 + 6 - wn.editCost * 12;
      if (!bestUnprovDacKhu || score > bestUnprovDacKhu.score) {
        bestUnprovDacKhu = {
          type: "ward_province_2025",
          ward: wn,
          province: null,
          widN: wn.id,
          pidN: dkRows[0][0],
          totalEdits: wn.editCost,
          score,
        };
      }
    }
    if (bestUnprovDacKhu) return bestUnprovDacKhu;

    // 5c. Unprovinced island district (e.g. 'Huyện Lý Sơn', 'Huyện Côn Đảo', 'Huyện Phú Quý')
    let bestUnprovIsle = null;
    for (const d of districts) {
      if (!d.prefixed && d.wordCount < 2) continue;
      const isleHits = (c.isle || []).filter((x) => x[0] === d.id);
      if (isleHits.length !== 1) continue;
      const hasSameWard = !d.prefixed && wardsOld.some((w) => w.start === d.start && w.end === d.end);
      if (hasSameWard) continue;
      const hit = isleHits[0];
      const score = (d.end - d.start) * 4 + (d.prefixed ? 4 : 0) - d.editCost * 12;
      if (!bestUnprovIsle || score > bestUnprovIsle.score) {
        bestUnprovIsle = {
          type: "island_special",
          district: d,
          province: null,
          didO: hit[0],
          pidO: hit[1],
          widN: hit[2],
          pidN: hit[3],
          totalEdits: d.editCost,
          score,
        };
      }
    }
    if (bestUnprovIsle) return bestUnprovIsle;

    // 6. Single-level fallback (for suggestions / partial recovery)
    let bestSingleW = null;
    for (const w of wardsOld) {
      if (!w.prefixed && w.wordCount < 2 && w.editCost > 0) continue;
      const score = (w.end - w.start) * 4 + (w.prefixed ? 3 : 0) - w.editCost * 10;
      if (!bestSingleW || score > bestSingleW.score) {
        bestSingleW = { ...w, score };
      }
    }
    if (bestSingleW) {
      return { type: "single_ward", ward: bestSingleW, widO: bestSingleW.id, totalEdits: bestSingleW.editCost };
    }
    return null;
  }

  function extractStreetPrefixRaw(rawText, norm, matchFirstTokenStart, tokens) {
    if (!rawText || matchFirstTokenStart <= 0) return "";
    // Try finding comma-separated segment before the matched phrase in rawText
    const parts = rawText.split(",").map((p) => p.trim()).filter(Boolean);
    if (parts.length >= 2) {
      // Check how many leading parts are outside the admin match
      const prefixNormTokens = tokens.slice(0, matchFirstTokenStart).filter((t) => t !== ",");
      if (prefixNormTokens.length > 0) {
        // Reconstruct prefix by taking comma chunks up to prefixNormTokens length
        const chosen = [];
        let tokCount = 0;
        for (let i = 0; i < parts.length - 1; i++) {
          const pNorm = tokenizePhraseStream(normalizeText(parts[i])).filter((t) => t !== ",");
          if (tokCount + pNorm.length <= prefixNormTokens.length) {
            chosen.push(parts[i]);
            tokCount += pNorm.length;
          } else {
            break;
          }
        }
        if (chosen.length > 0 && tokCount === prefixNormTokens.length) {
          return chosen.join(", ");
        }
      }
    }
    // Fallback: slice tokens before matchFirstTokenStart
    const slice = tokens.slice(0, matchFirstTokenStart).filter((t) => t !== ",");
    if (!slice.length) return "";
    // Avoid returning bare admin prefixes
    if (slice.every((t) => WARD_PREFIX_SET.has(t) || DIST_PREFIX_SET.has(t))) return "";
    return slice.join(" ");
  }

  function buildRepairedNormText(tokens, match) {
    if (!match) return tokens.filter((t) => t !== ",").join(" ");
    const out = tokens.slice();
    const replaceSpan = (span, replacement) => {
      if (!span) return;
      const repTokens = [];
      if (span.prefixed && span.rawPrefix) repTokens.push(span.rawPrefix);
      repTokens.push(...replacement.split(" "));
      for (let k = span.start; k < span.end; k++) {
        out[k] = k === span.start ? repTokens.join(" ") : "";
      }
    };
    if (match.ward) replaceSpan(match.ward, match.ward.name);
    if (match.district) replaceSpan(match.district, match.district.name);
    if (match.province) replaceSpan(match.province, match.province.name);
    return out.filter((t) => t !== "").join(" ").replace(/\s+,\s+/g, ", ").replace(/\s+/g, " ").trim();
  }

  /**
   * Returns repaired normalized text + exact/1-edit recovered (ward, district, province).
   */
  function fuzzyMatchGazetteer(text, opts) {
    ensureConverterLoaded();
    const raw = (text || "").trim();
    const repaired = repairTelexTypos(raw);
    const norm = normalizeText(repaired);
    const allowFuzzy = !opts || opts.allowFuzzy !== false;

    if (!state.initialized || !norm) {
      return {
        raw_text: raw,
        repaired_text: repaired,
        normalized_text: norm,
        fuzzy_repaired_text: norm,
        matched: false,
        match_type: null,
        edit_distance: 0,
        ward: null,
        district: null,
        province: null,
        formatted_legacy: null,
        formatted_2025: null,
        legacy_ids: null,
        new_ids: null,
        conversion_2025: null,
      };
    }

    const tokens = tokenizePhraseStream(norm);
    // Fast exact pass first; only run 1-edit fuzzy scan if exact pass didn't resolve a 2-level hierarchy
    let spans = scanSpans(tokens, false);
    let best = resolveBestHierarchy(spans);

    const hasResolvedPair =
      best &&
      best.type !== "single_ward";

    if (allowFuzzy && !hasResolvedPair) {
      spans = scanSpans(tokens, true);
      const fuzzyBest = resolveBestHierarchy(spans);
      if (fuzzyBest && (!best || fuzzyBest.type !== "single_ward")) {
        best = fuzzyBest;
      }
    }

    if (!best) {
      return {
        raw_text: raw,
        repaired_text: repaired,
        normalized_text: norm,
        fuzzy_repaired_text: norm,
        matched: false,
        match_type: null,
        edit_distance: 0,
        ward: null,
        district: null,
        province: null,
        formatted_legacy: null,
        formatted_2025: null,
        legacy_ids: null,
        new_ids: null,
        conversion_2025: null,
      };
    }

    const firstSpan = best.ward || best.district || best.province;
    const streetPrefix = extractStreetPrefixRaw(repaired, norm, firstSpan ? firstSpan.start : 0, tokens);
    const rawLower = repaired.toLowerCase();

    let conv = null;
    let wardDisp = null;
    let distDisp = null;
    let provDisp = null;
    let formattedLegacy = null;
    let formatted2025 = null;
    let newIds = null;
    let legacyIds = null;

    if (best.type === "ward_district_province" || best.type === "ward_district" || best.type === "ward_province") {
      conv = buildConversionFromEdge(
        best.widO,
        best.didO,
        best.pidO,
        streetPrefix,
        best.ward.rawPrefix,
        best.district ? best.district.rawPrefix : null,
        rawLower
      );
      if (conv) {
        wardDisp = conv.old_ward;
        distDisp = conv.old_district;
        provDisp = conv.old_province;
        formattedLegacy = conv.old_formatted;
        formatted2025 = conv.new_admin_formatted;
        const primaryEdge = (state.c.eo[best.widO] || []).find((e) => e[0] === best.didO && e[1] === best.pidO);
        if (primaryEdge) newIds = { wid_n: primaryEdge[2], pid_n: primaryEdge[3] };
      }
      legacyIds = { wid_o: best.widO, did_o: best.didO, pid_o: best.pidO };
    } else if (best.type === "ward_province_2025") {
      const nw = formatWard2025(best.widN, best.pidN);
      const np = state.c.pn_disp[best.pidN];
      wardDisp = nw;
      provDisp = np;
      formatted2025 = `${nw}, ${np}`;
      const enEntry = (state.c.en[best.widN] || []).find((x) => x[0] === best.pidN) || (state.c.en[best.widN] || [])[0];
      conv = {
        converted: false,
        already_2025: true,
        old_formatted: null,
        new_formatted: streetPrefix ? `${streetPrefix}, ${formatted2025}` : formatted2025,
        new_admin_formatted: formatted2025,
        new_ward: nw,
        new_province: np,
        is_divided: false,
        candidates: [{
          new_ward: nw,
          new_province: np,
          new_formatted: streetPrefix ? `${streetPrefix}, ${formatted2025}` : formatted2025,
          new_admin_formatted: formatted2025,
          is_default: true,
          lat: enEntry ? enEntry[1] : null,
          lon: enEntry ? enEntry[2] : null,
          area_km2: enEntry ? enEntry[3] : null,
          ward_code: enEntry ? enEntry[4] : null,
        }],
      };
      newIds = { wid_n: best.widN, pid_n: best.pidN };
    } else if (best.type === "island_special") {
      const nw = formatWard2025(best.widN, best.pidN);
      const np = state.c.pn_disp[best.pidN];
      distDisp = formatOldDistrict(best.didO, best.district.rawPrefix);
      provDisp = best.pidO !== undefined && best.pidO !== null ? state.c.po_disp[best.pidO] : null;
      formattedLegacy = provDisp ? `${distDisp}, ${provDisp}` : distDisp;
      formatted2025 = `${nw}, ${np}`;
      const enEntry = (state.c.en[best.widN] || []).find((x) => x[0] === best.pidN) || (state.c.en[best.widN] || [])[0];
      conv = {
        converted: true,
        already_2025: false,
        old_formatted: formattedLegacy,
        new_formatted: streetPrefix ? `${streetPrefix}, ${formatted2025}` : formatted2025,
        new_admin_formatted: formatted2025,
        new_ward: nw,
        new_province: np,
        is_divided: false,
        candidates: [{
          new_ward: nw,
          new_province: np,
          new_formatted: streetPrefix ? `${streetPrefix}, ${formatted2025}` : formatted2025,
          new_admin_formatted: formatted2025,
          is_default: true,
          lat: enEntry ? enEntry[1] : null,
          lon: enEntry ? enEntry[2] : null,
          area_km2: enEntry ? enEntry[3] : null,
          ward_code: enEntry ? enEntry[4] : null,
        }],
      };
      legacyIds = { wid_o: null, did_o: best.didO, pid_o: best.pidO };
      newIds = { wid_n: best.widN, pid_n: best.pidN };
    } else if (best.type === "district_province") {
      distDisp = formatOldDistrict(best.didO, best.district.rawPrefix);
      provDisp = state.c.po_disp[best.pidO];
      formattedLegacy = `${distDisp}, ${provDisp}`;
      legacyIds = { wid_o: null, did_o: best.didO, pid_o: best.pidO };
    } else if (best.type === "single_ward") {
      wardDisp = formatOldWard(best.widO, null, null, best.ward.rawPrefix, rawLower);
      legacyIds = { wid_o: best.widO, did_o: null, pid_o: null };
    }

    return {
      raw_text: raw,
      repaired_text: repaired,
      normalized_text: norm,
      fuzzy_repaired_text: buildRepairedNormText(tokens, best),
      matched: true,
      match_type: best.type,
      edit_distance: best.totalEdits || 0,
      ward: wardDisp,
      district: distDisp,
      province: provDisp,
      formatted_legacy: formattedLegacy,
      formatted_2025: formatted2025,
      legacy_ids: legacyIds,
      new_ids: newIds,
      conversion_2025: conv,
    };
  }

  /**
   * Converts a raw address string or `{ ward, district, province }` object into canonical 2025 2-tier format.
   */
  function convertLegacyTo2025(input, precomputedMatch) {
    ensureConverterLoaded();
    if (!state.initialized) {
      return {
        converted: false,
        old_formatted: null,
        new_formatted: null,
        new_ward: null,
        new_province: null,
        is_divided: false,
        candidates: [],
      };
    }

    let text = "";
    if (input && typeof input === "object" && !Array.isArray(input)) {
      const parts = [input.street, input.ward, input.district, input.province].filter(Boolean);
      text = parts.join(", ");
    } else {
      text = String(input || "");
    }

    const match = precomputedMatch || fuzzyMatchGazetteer(text);
    if (match && match.conversion_2025) {
      return match.conversion_2025;
    }

    // If only a single ward matched, surface its candidate conversions without claiming converted=true
    if (match && match.legacy_ids && match.legacy_ids.wid_o !== null) {
      const edges = state.c.eo[match.legacy_ids.wid_o] || [];
      const seen = new Set();
      const candidates = [];
      for (const e of edges) {
        const key = `${e[2]}:${e[3]}`;
        if (seen.has(key)) continue;
        seen.add(key);
        const nw = formatWard2025(e[2], e[3]);
        const np = state.c.pn_disp[e[3]];
        candidates.push({
          new_ward: nw,
          new_province: np,
          new_formatted: `${nw}, ${np}`,
          old_district: formatOldDistrict(e[0], null),
          old_province: state.c.po_disp[e[1]],
          is_default: e[5] === 1,
        });
      }
      return {
        converted: false,
        old_formatted: match.ward || null,
        new_formatted: null,
        new_ward: null,
        new_province: null,
        is_divided: false,
        candidates,
      };
    }

    return {
      converted: false,
      old_formatted: null,
      new_formatted: null,
      new_ward: null,
      new_province: null,
      is_divided: false,
      candidates: [],
    };
  }

  /**
   * Returns Top-K clickable ward+province suggestions for incomplete or typo address inputs.
   */
  function suggestWards(text, limit = 3) {
    ensureConverterLoaded();
    if (!state.initialized || !text || !String(text).trim()) return [];
    const c = state.c;
    const raw = String(text).trim();
    const repaired = repairTelexTypos(raw);
    const norm = normalizeText(repaired);
    if (!norm) return [];

    const rawLower = repaired.toLowerCase();
    const tokens = tokenizePhraseStream(norm);
    let spans = scanSpans(tokens, false);
    if (spans.wardsOld.length === 0 && spans.wardsNew.length === 0 && spans.districts.length === 0) {
      spans = scanSpans(tokens, true);
    }
    const results = [];
    const seen = new Set();

    // Check if input resolves cleanly as a District + Province (missing ward), e.g. "Thanh Xuân, Hà Nội"
    let matchedDP = null;
    if (spans.districts.length > 0 && spans.provincesOld.length > 0) {
      for (const d of spans.districts) {
        const dpPids = c.dp[d.id] || [];
        for (const p of spans.provincesOld) {
          if (disjoint(d, p) && d.start <= p.start && dpPids.includes(p.id)) {
            matchedDP = { d, p };
            break;
          }
        }
        if (matchedDP) break;
      }
    }
    // Check if there's an unshadowed ward+district or ward+province match first
    const hasWardParentPair = spans.wardsOld.some((w) => {
      const edges = c.eo[w.id] || [];
      return spans.districts.some((d) => disjoint(w, d) && w.start <= d.start && edges.some((e) => e[0] === d.id)) ||
        spans.provincesOld.some((p) => {
          if (!disjoint(w, p) || w.start > p.start || !edges.some((e) => e[1] === p.id)) return false;
          if (!w.prefixed && matchedDP && !(w.end <= matchedDP.d.start || matchedDP.d.end <= w.start)) return false;
          return true;
        });
    });

    // Case A: User provided ward name (exact or 1-edit) — e.g. "Số 5, Lộc Thạnh", "Xã Tân Lập", "loc tahnh"
    if ((!matchedDP || hasWardParentPair) && (spans.wardsOld.length > 0 || spans.wardsNew.length > 0)) {
      const scored = [];
      const distHints = new Set(spans.districts.map((d) => d.id));
      const provOldHints = new Set(spans.provincesOld.map((p) => p.id));
      const provNewHints = new Set(spans.provincesNew.map((p) => p.id));

      for (const w of spans.wardsOld) {
        if (!w.prefixed && w.wordCount < 2 && w.editCost > 0) continue;
        const edges = c.eo[w.id] || [];
        const streetPrefix = extractStreetPrefixRaw(repaired, norm, w.start, tokens);

        for (const e of edges) {
          const [didO, pidO, widN, pidN, , isDef] = e;
          const nw = formatWard2025(widN, pidN);
          const np = c.pn_disp[pidN];
          const oldW = formatOldWard(w.id, widN, pidN, w.rawPrefix, rawLower);
          const oldD = formatOldDistrict(didO, null);
          const oldDShort = shortDistrictLabel(didO, null);
          const oldP = c.po_disp[pidO];
          const wnBare = c.wn_disp[widN];
          const enEntry = (c.en[widN] || []).find((x) => x[0] === pidN) || (c.en[widN] || [])[0];
          const wtNew = enEntry ? c.wt[enEntry[5]] : "Xã";

          const sameWardName = normalizeText(nw) === normalizeText(oldW);
          let score = (w.end - w.start) * 10 - w.editCost * 25;
          if (w.prefixed) score += 6;
          if (isDef) score += 3;
          if (sameWardName) score += 4;
          if (distHints.has(didO)) score += 30;
          if (provOldHints.has(pidO) || provNewHints.has(pidN)) score += 20;
          // Boost exact accented match (crucial for "Lộc Thạnh" vs "Lộc Thanh" vs "Lộc Thành")
          if (wnBare && rawLower.includes(wnBare.toLowerCase())) score += 45;
          // Boost matching ward type prefix if user typed "Xã" / "Phường"
          if (w.rawPrefix === "xa" && wtNew === "Xã") score += 12;
          if (w.rawPrefix === "phuong" && wtNew === "Phường") score += 12;

          const provDisplay = formatProvinceWithLevel(np, c.pn[pidN]);
          const legacyDetail = sameWardName
            ? `${oldDShort}, ${oldP}`
            : `${oldW.replace(/^Phường /, "P. ").replace(/^Xã /, "X. ")}, ${oldDShort}, ${oldP}`;
          const label = `${nw}, ${provDisplay} (cũ: ${legacyDetail})`;
          const insertText = streetPrefix ? `${streetPrefix}, ${nw}, ${np}` : `${nw}, ${np}`;

          scored.push({
            score,
            dedupeKey: `${widN}:${pidN}:${didO}`,
            item: {
              label,
              insert_text: insertText,
              new_ward: nw,
              new_province: np,
              old_ward: oldW,
              old_district: oldD,
              old_province: oldP,
            },
          });
        }
      }

      scored.sort((a, b) => b.score - a.score);
      for (const s of scored) {
        if (seen.has(s.dedupeKey)) continue;
        seen.add(s.dedupeKey);
        results.push(s.item);
        if (results.length >= limit) return results;
      }
    }

    // Case B: User provided District + Province without Ward (e.g. "Thanh Xuân, Hà Nội")
    if (results.length < limit && spans.districts.length > 0) {
      const targetD = spans.districts[0];
      const allowedPids = spans.provincesOld.length > 0
        ? new Set(spans.provincesOld.map((p) => p.id))
        : new Set(c.dp[targetD.id] || []);
      const streetPrefix = extractStreetPrefixRaw(repaired, norm, targetD.start, tokens);

      for (let widO = 0; widO < c.eo.length && results.length < limit; widO++) {
        const edges = c.eo[widO];
        for (let j = 0; j < edges.length && results.length < limit; j++) {
          const [didO, pidO, widN, pidN] = edges[j];
          if (didO !== targetD.id || !allowedPids.has(pidO)) continue;
          const dk = `${widN}:${pidN}:${didO}`;
          if (seen.has(dk)) continue;
          seen.add(dk);

          const nw = formatWard2025(widN, pidN);
          const np = c.pn_disp[pidN];
          const oldW = formatOldWard(widO, widN, pidN, null, rawLower);
          const oldD = formatOldDistrict(didO, targetD.rawPrefix);
          const oldDShort = shortDistrictLabel(didO, targetD.rawPrefix);
          const oldP = c.po_disp[pidO];
          const provDisplay = formatProvinceWithLevel(np, c.pn[pidN]);
          const label = `${nw}, ${provDisplay} (cũ: ${oldW}, ${oldDShort}, ${oldP})`;
          const insertText = streetPrefix ? `${streetPrefix}, ${nw}, ${np}` : `${nw}, ${np}`;
          results.push({
            label,
            insert_text: insertText,
            new_ward: nw,
            new_province: np,
            old_ward: oldW,
            old_district: oldD,
            old_province: oldP,
          });
        }
      }
    }

    // Case C: Prefix autocomplete fallback when user stopped mid-typing (e.g. "loc tha")
    if (results.length === 0) {
      const cleanTokens = tokens.filter((t) => t !== "," && !WARD_PREFIX_SET.has(t));
      const tail = cleanTokens.slice(-2).join(" ");
      if (tail.length >= 4 && !/^\d+$/.test(tail) && !cleanTokens.some((w) => GRANULAR_SKIP_WORDS.has(w))) {
        for (let widO = 0; widO < c.wo.length && results.length < limit; widO++) {
          if (!c.wo[widO].startsWith(tail)) continue;
          for (const [didO, pidO, widN, pidN] of c.eo[widO]) {
            const dk = `${widN}:${pidN}:${didO}`;
            if (seen.has(dk)) continue;
            seen.add(dk);
            const nw = formatWard2025(widN, pidN);
            const np = c.pn_disp[pidN];
            const oldW = formatOldWard(widO, widN, pidN, null, rawLower);
            const oldD = formatOldDistrict(didO, null);
            const oldDShort = shortDistrictLabel(didO, null);
            const oldP = c.po_disp[pidO];
            const provDisplay = formatProvinceWithLevel(np, c.pn[pidN]);
            results.push({
              label: `${nw}, ${provDisplay} (cũ: ${oldDShort}, ${oldP})`,
              insert_text: `${nw}, ${np}`,
              new_ward: nw,
              new_province: np,
              old_ward: oldW,
              old_district: oldD,
              old_province: oldP,
            });
            if (results.length >= limit) break;
          }
        }
      }
    }

    return results.slice(0, limit);
  }

  const api = {
    stripDiacritics,
    normalizeText,
    repairTelexTypos,
    initConverterV3,
    isConverterInitialized,
    fuzzyMatchGazetteer,
    convertLegacyTo2025,
    suggestWards,
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    root.LayaTelexFuzzy = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
