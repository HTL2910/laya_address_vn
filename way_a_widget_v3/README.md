# @laya/address-vn-widget (v3.0.0)

Zero-CDN, 100% client-side Vietnamese address validation gate, Telex/VNI & 1-edit fuzzy recovery engine, Legacy-to-2025 2-tier administrative converter, and Form-Associated `<laya-address-input>` web component.

## Features

- **Zero CDN / Zero Server Inference Cost**: All runtime scripts, `onnxruntime-web` WASM binaries, Char-CNN v2 ONNX model (`1.12 MB`), GSO/2025 gazetteer (`vn_admin_gazetteer.json`), and v3 converter (`vn_admin_v3_converter.json`) load strictly from local `dist/` and `sdk/`.
- **Telex & VNI Typo Repair (`repairTelexTypos`)**: Repairs stuck-key / IME-off typos (`phuowng -> phuong`, `loocj -> loc`, `binhf -> binh`, `thanhf -> thanh`, `huyeenj -> huyen`, `quaanj -> quan`, `dduowngf -> duong`, `xaax -> xa`, `haf nooij -> ha noi`) while preserving URLs, emails, house numbers/slashes (`45/2B`), phone numbers, and common English words (`ok`, `shop`, `jean`, `bbq`, `size`).
- **SymSpell 1-Edit Fuzzy Gazetteer (`fuzzyMatchGazetteer`)**: Sub-25ms init over 63 old provinces, 661 districts, 6,433 old wards, 34 new provinces, and 2,745 new 2025 wards; recovers 1-edit omissions, insertions, substitutions, and adjacent transpositions (`loc tahnh, loc ninh` -> `Xã Lộc Thạnh, Huyện Lộc Ninh, Bình Phước`).
- **Legacy -> 2025 2-Tier Converter (`convertLegacyTo2025`)**: Converts legacy 3-tier addresses (e.g., `12 Nguyễn Trãi, Phúc Xá, Ba Đình, Hà Nội`) into canonical 2025 2-tier format (`12 Nguyễn Trãi, Phường Hồng Hà, Hà Nội`), including divided-ward candidates and island special zones (`Đặc khu`).
- **Top-3 1-Click Suggestions (`suggestWards`)**: Returns actionable Top-3 ward + province suggestions with legacy provenance labels and autofill `insert_text`.
- **Drop-In Integration**: Works via `LayaAddressSDK.attachToInput(inputElement)` on Webcake/Shopify/Haravan forms OR as `<laya-address-input>` Form-Associated Custom Element, plus UMD `require('@laya/address-vn-widget')` in Node.js.

## Browser Usage (Zero CDN)

```html
<script src="../dist/vendor/onnxruntime-web/ort.wasm.min.js" defer></script>
<script src="../dist/heuristics.js" defer></script>
<script src="../sdk/laya-telex-fuzzy.js" defer></script>
<script src="../sdk/laya-address-widget.js" defer></script>
<script>
  window.addEventListener("DOMContentLoaded", async () => {
    await window.LayaAddressSDK.init({ basePath: "../dist" });
    window.LayaAddressSDK.attachToInput(document.querySelector("#address"));
  });
</script>
```

Or drop in `<laya-address-input name="address" base-path="../dist" required></laya-address-input>` directly inside any `<form>`.

## Node.js Usage

```js
const LayaAddressSDK = require("@laya/address-vn-widget");

await LayaAddressSDK.init();
const result = LayaAddressSDK.evaluate("12 Nguyễn Trãi, Phúc Xá, Ba Đình, Hà Nội");
// result.sufficiency === "address_sufficient"
// result.conversion_2025.new_ward === "Phường Hồng Hà"
```

## Development & Verification

```powershell
node --check sdk\laya-telex-fuzzy.js
node --check sdk\laya-address-widget.js
node --test tests\test_widget_v3.mjs
python serve.py --port 8086
```

## License & Data Attribution

- Code & Widget SDK: MIT License
- Administrative unit conversion & 2025 dataset derived from `tranngocminhhieu/vietnamadminunits` (MIT License — see `LICENSE_vietnamadminunits.txt`).
