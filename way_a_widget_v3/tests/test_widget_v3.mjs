import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT_DIR = path.resolve(__dirname, "..");

const TelexFuzzy = require(path.join(ROOT_DIR, "sdk", "laya-telex-fuzzy.js"));
const LayaAddressSDK = require(path.join(ROOT_DIR, "sdk", "laya-address-widget.js"));

const converterJson = JSON.parse(
  fs.readFileSync(path.join(ROOT_DIR, "dist", "model", "vn_admin_v3_converter.json"), "utf8")
);

await LayaAddressSDK.init();

test("initConverterV3 builds 1-edit index under 25 ms", () => {
  // Warm call ensures JIT overhead isn't skewing index build cost
  const t0 = performance.now();
  const ok = TelexFuzzy.initConverterV3(converterJson);
  const dt = performance.now() - t0;
  assert.equal(ok, true);
  assert.equal(TelexFuzzy.isConverterInitialized(), true);
  assert.ok(dt < 25, `Expected initConverterV3 < 25 ms, got ${dt.toFixed(2)} ms`);
});

test("repairTelexTypos fixes Vietnamese Telex & VNI typos accurately", () => {
  assert.equal(TelexFuzzy.repairTelexTypos("phuowng"), "phuong");
  assert.equal(TelexFuzzy.repairTelexTypos("loocj"), "loc");
  assert.equal(TelexFuzzy.repairTelexTypos("binhf"), "binh");
  assert.equal(TelexFuzzy.repairTelexTypos("thanhf"), "thanh");
  assert.equal(TelexFuzzy.repairTelexTypos("huyeenj"), "huyen");
  assert.equal(TelexFuzzy.repairTelexTypos("quaanj"), "quan");
  assert.equal(TelexFuzzy.repairTelexTypos("dduowngf"), "duong");
  assert.equal(TelexFuzzy.repairTelexTypos("xaax"), "xa");
  assert.equal(TelexFuzzy.repairTelexTypos("haf nooij"), "ha noi");
  assert.equal(
    TelexFuzzy.repairTelexTypos("xa3 Lo6c5 Tha5nh, Huye6n5 Lo6c5 Ninh2"),
    "xa Loc Thanh, Huyen Loc Ninh"
  );
});

test("repairTelexTypos preserves URLs, emails, house slashes, phones, and English words", () => {
  const sample =
    "ok shop jean bbq size https://shop.vn/phuowng-binhf user.loocj@gmail.com 45/2B 0912345678 +84 987 654 321";
  assert.equal(TelexFuzzy.repairTelexTypos(sample), sample);
});

test("Telex typo evaluation resolves Xã Loocj Thanhj, H. Loocj Ninhf to sufficient + 2025 Đồng Nai", () => {
  const res = LayaAddressSDK.evaluate("Xã Loocj Thanhj, H. Loocj Ninhf");
  assert.equal(res.is_address, true);
  assert.equal(res.sufficiency, "address_sufficient");
  assert.equal(res.repaired_text, "Xã Loc Thanh, H. Loc Ninh");
  assert.ok(res.conversion_2025);
  assert.equal(res.conversion_2025.new_ward, "Xã Lộc Thạnh");
  assert.equal(res.conversion_2025.new_province, "Đồng Nai");
});

test("1-edit typo recovery resolves loc tahnh, loc ninh as valid sufficient", () => {
  const fuzzy = TelexFuzzy.fuzzyMatchGazetteer("loc tahnh, loc ninh");
  assert.equal(fuzzy.matched, true);
  assert.equal(fuzzy.edit_distance, 1);
  assert.equal(fuzzy.formatted_legacy, "Xã Lộc Thạnh, Huyện Lộc Ninh, Bình Phước");

  const res = LayaAddressSDK.evaluate("loc tahnh, loc ninh");
  assert.equal(res.is_address, true);
  assert.equal(res.sufficiency, "address_sufficient");
  assert.equal(res.conversion_2025.new_ward, "Xã Lộc Thạnh");
  assert.equal(res.conversion_2025.new_province, "Đồng Nai");
});

test("Numeric ward/district codes never fuzzy-match wrong numbers", () => {
  const res = LayaAddressSDK.evaluate("Phường 99, Quận 99, Hồ Chí Minh");
  assert.notEqual(res.conversion_2025?.new_ward, "Phường 9");
});

test("Legacy -> 2025 live conversion maps 12 Nguyễn Trãi, Phúc Xá, Ba Đình, Hà Nội", () => {
  const conv = TelexFuzzy.convertLegacyTo2025("12 Nguyễn Trãi, Phúc Xá, Ba Đình, Hà Nội");
  assert.equal(conv.converted, true);
  assert.equal(conv.new_ward, "Phường Hồng Hà");
  assert.equal(conv.new_province, "Hà Nội");
  assert.equal(conv.new_formatted, "12 Nguyễn Trãi, Phường Hồng Hà, Hà Nội");

  const res = LayaAddressSDK.evaluate("12 Nguyễn Trãi, Phúc Xá, Ba Đình, Hà Nội");
  assert.equal(res.is_address, true);
  assert.equal(res.sufficiency, "address_sufficient");
  assert.equal(res.conversion_2025.new_ward, "Phường Hồng Hà");
  assert.equal(res.conversion_2025.new_province, "Hà Nội");
});

test("Divided wards and island special zones convert cleanly", () => {
  const divided = TelexFuzzy.convertLegacyTo2025("Phường Cống Vị, Quận Ba Đình, Hà Nội");
  assert.equal(divided.converted, true);
  assert.equal(divided.is_divided, true);
  assert.ok(divided.candidates.length >= 2);

  const island = LayaAddressSDK.evaluate("Côn Đảo, Bà Rịa - Vũng Tàu");
  assert.equal(island.is_address, true);
  assert.equal(island.sufficiency, "address_sufficient");
  assert.equal(island.conversion_2025.new_ward, "Đặc khu Côn Đảo");
  assert.equal(island.conversion_2025.new_province, "Hồ Chí Minh");
});

test("Core sufficiency cases: sufficient, insufficient, and not_address", () => {
  const suf = LayaAddressSDK.evaluate("Lộc Thạnh, Lộc Ninh");
  assert.equal(suf.sufficiency, "address_sufficient");
  assert.equal(suf.is_address, true);

  const ins = LayaAddressSDK.evaluate("ngõ 12");
  assert.equal(ins.sufficiency, "address_insufficient");
  assert.equal(ins.is_address, false);
  assert.ok(ins.missing_field_hint && ins.missing_field_hint.length > 0);

  const bbq = LayaAddressSDK.evaluate("Cho mình 3 chai nướng BBQ");
  assert.equal(bbq.sufficiency, "not_address");
  assert.equal(bbq.is_address, false);

  const person = LayaAddressSDK.evaluate("Nguyễn Thị Thanh Xuân");
  assert.equal(person.sufficiency, "not_address");
  assert.equal(person.is_address, false);
});

test("suggestWards returns Top-3 clickable suggestions with 2025 new_ward and new_province", () => {
  const sugs = TelexFuzzy.suggestWards("Số 5, Tân Lập", 3);
  assert.equal(sugs.length, 3);
  for (const s of sugs) {
    assert.ok(s.label && s.label.length > 5);
    assert.ok(s.insert_text && s.insert_text.startsWith("Số 5, "));
    assert.ok(s.new_ward && s.new_province);
    assert.ok(s.old_ward && s.old_district && s.old_province);
    // Clicking suggestion and re-evaluating must produce address_sufficient!
    const reEval = LayaAddressSDK.evaluate(s.insert_text);
    assert.equal(reEval.sufficiency, "address_sufficient");
    assert.equal(reEval.is_address, true);
  }

  const locSugs = TelexFuzzy.suggestWards("Số 5, Lộc Thạnh", 3);
  assert.equal(locSugs.length, 3);
  assert.equal(locSugs[0].new_ward, "Xã Lộc Thạnh");
  assert.equal(locSugs[0].new_province, "Đồng Nai");

  const typoSugs = TelexFuzzy.suggestWards("loc tahnh", 3);
  assert.equal(typoSugs.length, 3);
  assert.equal(typoSugs[0].new_ward, "Xã Lộc Thạnh");
});

test("Zero external CDN links in demo/index.html and sdk/*.js", () => {
  const filesToScan = [
    path.join(ROOT_DIR, "demo", "index.html"),
    path.join(ROOT_DIR, "sdk", "laya-telex-fuzzy.js"),
    path.join(ROOT_DIR, "sdk", "laya-address-widget.js"),
  ];
  const externalSchemeRe = /https?:\/\//i;
  for (const fp of filesToScan) {
    const content = fs.readFileSync(fp, "utf8");
    assert.equal(
      externalSchemeRe.test(content),
      false,
      `Found external http(s):// reference in ${path.relative(ROOT_DIR, fp)}`
    );
  }
});

test("Required dist assets and package.json metadata exist", () => {
  const pkg = JSON.parse(fs.readFileSync(path.join(ROOT_DIR, "package.json"), "utf8"));
  assert.equal(pkg.name, "@laya/address-vn-widget");
  assert.equal(pkg.version, "3.0.0");
  assert.equal(pkg.license, "MIT");

  const requiredFiles = [
    "LICENSE_vietnamadminunits.txt",
    "dist/heuristics.js",
    "dist/model/laya_address_vn.onnx",
    "dist/model/vocab.json",
    "dist/model/calibration.json",
    "dist/model/vn_admin_gazetteer.json",
    "dist/model/vn_admin_v3_converter.json",
    "dist/vendor/onnxruntime-web/ort.wasm.min.js",
    "dist/vendor/onnxruntime-web/ort-wasm-simd-threaded.wasm",
  ];
  for (const rel of requiredFiles) {
    assert.ok(fs.existsSync(path.join(ROOT_DIR, rel)), `Missing required asset: ${rel}`);
  }
});

test("Average evaluation latency < 1.0 ms across 500 mixed evaluations", () => {
  const samples = [
    "Xã Loocj Thanhj, H. Loocj Ninhf",
    "loc tahnh, loc ninh",
    "12 Nguyễn Trãi, Phúc Xá, Ba Đình, Hà Nội",
    "Lộc Thạnh, Lộc Ninh",
    "ngõ 12",
    "Cho mình 3 chai nướng BBQ",
    "Số 5, Tân Lập",
    "Phường Hồng Hà, Hà Nội",
    "Thanh Xuân, Hà Nội",
    "Nguyễn Thị Thanh Xuân",
  ];
  // Warmup once
  for (const s of samples) LayaAddressSDK.evaluate(s);

  const t0 = performance.now();
  for (let i = 0; i < 500; i++) {
    const r = LayaAddressSDK.evaluate(samples[i % samples.length]);
    assert.ok(r && typeof r.sufficiency === "string");
  }
  const totalMs = performance.now() - t0;
  const avgMs = totalMs / 500;
  console.log(`  [perf] 500 evaluations: total=${totalMs.toFixed(2)} ms, avg=${avgMs.toFixed(4)} ms`);
  assert.ok(avgMs < 1.0, `Expected average latency < 1.0 ms, got ${avgMs.toFixed(4)} ms`);
});
