#!/usr/bin/env python3
"""Create an interactive HTML creator to visualize bundle/product CSV uploads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PRODUCTS_CSV = Path("data/product_dataset.csv")
PRODUCT_ID_COLUMN = "product_asset_id"
PRODUCT_DESCRIPTION_COLUMN = "product_description"
DEFAULT_OUTPUT = Path("outputs/bundle_csv_creator.html")

# Hardcoded image paths relative to the generated HTML in outputs/
BUNDLE_IMAGE_BASE = "../data/bundle_images/"
PRODUCT_IMAGE_BASE = "../data/product_images/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an HTML page with a centered upload button to load a bundles "
            "CSV (bundle_asset_id, product_asset_id) and render bundle/product images."
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_product_descriptions() -> dict[str, str]:
    if not PRODUCTS_CSV.exists():
        raise FileNotFoundError(f"Missing required file: {PRODUCTS_CSV}")

    df = pd.read_csv(
        PRODUCTS_CSV, usecols=[PRODUCT_ID_COLUMN, PRODUCT_DESCRIPTION_COLUMN]
    )
    df = df.dropna(subset=[PRODUCT_ID_COLUMN]).copy()
    df[PRODUCT_ID_COLUMN] = df[PRODUCT_ID_COLUMN].astype(str)
    df[PRODUCT_DESCRIPTION_COLUMN] = df[PRODUCT_DESCRIPTION_COLUMN].fillna("").astype(str)

    descriptions: dict[str, str] = {}
    for product_id, description in zip(df[PRODUCT_ID_COLUMN], df[PRODUCT_DESCRIPTION_COLUMN]):
        product_id = product_id.strip()
        description = description.strip()
        if product_id and description and product_id not in descriptions:
            descriptions[product_id] = description
    return descriptions


def build_html(product_descriptions: dict[str, str]) -> str:
    descriptions_json = json.dumps(product_descriptions, ensure_ascii=True)
    bundle_base = json.dumps(BUNDLE_IMAGE_BASE)
    product_base = json.dumps(PRODUCT_IMAGE_BASE)

    return f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Bundle Viewer</title>
  <style>
    :root {{
      --bg: #ffffff;
      --text: #111111;
      --muted: #606060;
      --line: #d8d8d8;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    }}

    #csv-input {{
      position: absolute;
      width: 0;
      height: 0;
      opacity: 0;
      pointer-events: none;
    }}

    .upload-screen {{
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #ffffff;
      padding: 24px;
    }}

    .upload-btn {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      user-select: none;
      text-transform: uppercase;
      letter-spacing: 0.18em;
      transition: background-color 0.2s ease, color 0.2s ease, transform 0.2s ease;
    }}

    .upload-main {{
      min-width: min(620px, 92vw);
      min-height: 94px;
      border: 1px solid #111111;
      background: #ffffff;
      color: #111111;
      font-size: clamp(13px, 2.2vw, 18px);
      font-weight: 500;
      padding: 0 24px;
    }}

    .upload-main:hover {{
      background: #111111;
      color: #ffffff;
      transform: translateY(-1px);
    }}

    #gallery-root {{
      display: none;
      max-width: 1240px;
      margin: 0 auto;
      padding: 24px 18px 56px;
    }}

    .top-bar {{
      position: sticky;
      top: 0;
      z-index: 30;
      display: grid;
      grid-template-columns: minmax(120px, auto) 1fr minmax(190px, auto);
      align-items: center;
      gap: 14px;
      padding: 14px 0 20px;
      background: rgba(255, 255, 255, 0.96);
      border-bottom: 1px solid var(--line);
      margin-bottom: 24px;
    }}

    .brand {{
      font-family: "Didot", "Bodoni MT", "Times New Roman", serif;
      text-transform: uppercase;
      letter-spacing: 0.2em;
      font-size: 19px;
    }}

    .summary {{
      text-align: center;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}

    .upload-secondary {{
      justify-self: end;
      border: 1px solid #111111;
      background: #ffffff;
      color: #111111;
      font-size: 10px;
      font-weight: 500;
      min-height: 40px;
      padding: 0 18px;
      letter-spacing: 0.16em;
    }}

    .upload-secondary:hover {{
      background: #111111;
      color: #ffffff;
    }}

    .gallery {{
      display: grid;
      gap: 54px;
    }}

    .bundle {{
      border-top: 1px solid #111111;
      padding-top: 28px;
    }}

    .bundle-head {{
      display: grid;
      justify-items: center;
      gap: 16px;
      margin-bottom: 28px;
      text-align: center;
    }}

    .bundle-image {{
      width: min(450px, 100%);
      aspect-ratio: 3 / 4;
      object-fit: cover;
      border: 1px solid #ececec;
      background: #f7f7f7;
    }}

    .bundle-title {{
      margin: 0;
      font-weight: 500;
      font-size: clamp(16px, 2vw, 21px);
      text-transform: uppercase;
      letter-spacing: 0.1em;
    }}

    .bundle-meta {{
      margin: 0;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.14em;
    }}

    .products {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 22px;
    }}

    .product-card {{
      display: grid;
      gap: 10px;
      align-content: start;
    }}

    .product-image {{
      width: 100%;
      aspect-ratio: 3 / 4;
      object-fit: cover;
      border: 1px solid #ececec;
      background: #f7f7f7;
    }}

    .product-type {{
      border: 1px solid #111111;
      padding: 6px 8px;
      text-align: center;
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      line-height: 1.35;
      min-height: 34px;
      display: flex;
      align-items: center;
      justify-content: center;
    }}

    .asset-id {{
      color: var(--muted);
      font-size: 10px;
      letter-spacing: 0.06em;
      text-align: center;
      word-break: break-all;
      text-transform: uppercase;
    }}

    .missing {{
      display: flex;
      align-items: center;
      justify-content: center;
      text-align: center;
      border: 1px dashed #b8b8b8;
      color: #6a6a6a;
      background: #fafafa;
      padding: 14px;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}

    @media (max-width: 840px) {{
      .top-bar {{
        grid-template-columns: 1fr;
        justify-items: center;
        gap: 10px;
        padding-top: 0;
      }}

      .upload-secondary {{
        justify-self: center;
      }}

      .summary {{
        order: 3;
      }}
    }}
  </style>
</head>
<body>
  <input id="csv-input" type="file" accept=".csv,text/csv">

  <section id="upload-screen" class="upload-screen">
    <label for="csv-input" class="upload-btn upload-main">Subir CSV de bundles</label>
  </section>

  <main id="gallery-root">
    <header class="top-bar">
      <div class="brand">Bundle Edit</div>
      <div id="summary" class="summary"></div>
      <label for="csv-input" class="upload-btn upload-secondary">Cambiar CSV</label>
    </header>
    <div id="gallery" class="gallery"></div>
  </main>

  <script>
    const BUNDLE_IMAGE_BASE = {bundle_base};
    const PRODUCT_IMAGE_BASE = {product_base};
    const PRODUCT_DESCRIPTIONS = {descriptions_json};

    const csvInput = document.getElementById("csv-input");
    const uploadScreen = document.getElementById("upload-screen");
    const galleryRoot = document.getElementById("gallery-root");
    const gallery = document.getElementById("gallery");
    const summary = document.getElementById("summary");

    function parseCsvLine(line) {{
      const out = [];
      let current = "";
      let inQuotes = false;
      for (let i = 0; i < line.length; i++) {{
        const ch = line[i];
        if (ch === '"') {{
          const next = line[i + 1];
          if (inQuotes && next === '"') {{
            current += '"';
            i++;
          }} else {{
            inQuotes = !inQuotes;
          }}
        }} else if (ch === "," && !inQuotes) {{
          out.push(current);
          current = "";
        }} else {{
          current += ch;
        }}
      }}
      out.push(current);
      return out.map((value) => value.trim());
    }}

    function parseCsvRows(text) {{
      const normalized = text.replace(/^\\uFEFF/, "").replace(/\\r\\n/g, "\\n").replace(/\\r/g, "\\n");
      const lines = normalized.split("\\n").filter((line) => line.trim().length > 0);
      if (lines.length < 2) {{
        throw new Error("CSV vacio o sin filas.");
      }}

      const header = parseCsvLine(lines[0]);
      const bundleIdx = header.indexOf("bundle_asset_id");
      const productIdx = header.indexOf("product_asset_id");

      if (bundleIdx === -1 || productIdx === -1) {{
        throw new Error("El CSV debe contener columnas: bundle_asset_id y product_asset_id.");
      }}

      const grouped = new Map();
      for (let i = 1; i < lines.length; i++) {{
        const fields = parseCsvLine(lines[i]);
        const bundleId = (fields[bundleIdx] || "").trim();
        const productId = (fields[productIdx] || "").trim();
        if (!bundleId || !productId) {{
          continue;
        }}
        if (!grouped.has(bundleId)) {{
          grouped.set(bundleId, new Set());
        }}
        grouped.get(bundleId).add(productId);
      }}
      return grouped;
    }}

    function createImageOrMissing(src, alt, className) {{
      const img = document.createElement("img");
      img.className = className;
      img.src = src;
      img.alt = alt;
      img.loading = "lazy";
      img.referrerPolicy = "no-referrer";
      img.addEventListener("error", () => {{
        const missing = document.createElement("div");
        missing.className = className + " missing";
        missing.textContent = "Imagen no disponible";
        img.replaceWith(missing);
      }});
      return img;
    }}

    function productTypeLabel(productId) {{
      const typeText = PRODUCT_DESCRIPTIONS[productId];
      if (!typeText) {{
        return "Tipo: no disponible";
      }}
      return "Tipo: " + typeText;
    }}

    function renderGrouped(grouped) {{
      gallery.innerHTML = "";
      const bundles = [...grouped.entries()].sort((left, right) => left[0].localeCompare(right[0]));

      let totalProducts = 0;
      const fragment = document.createDocumentFragment();

      for (const [bundleId, productSet] of bundles) {{
        const productIds = [...productSet].sort((left, right) => left.localeCompare(right));
        totalProducts += productIds.length;

        const section = document.createElement("section");
        section.className = "bundle";

        const bundleHead = document.createElement("div");
        bundleHead.className = "bundle-head";

        const bundleImage = createImageOrMissing(
          BUNDLE_IMAGE_BASE + encodeURIComponent(bundleId) + ".jpg",
          bundleId,
          "bundle-image"
        );
        bundleHead.appendChild(bundleImage);

        const title = document.createElement("h2");
        title.className = "bundle-title";
        title.textContent = "Bundle " + bundleId;
        bundleHead.appendChild(title);

        const meta = document.createElement("p");
        meta.className = "bundle-meta";
        meta.textContent = "Productos asociados: " + productIds.length;
        bundleHead.appendChild(meta);

        section.appendChild(bundleHead);

        const productsGrid = document.createElement("div");
        productsGrid.className = "products";

        for (const productId of productIds) {{
          const card = document.createElement("article");
          card.className = "product-card";

          const productImage = createImageOrMissing(
            PRODUCT_IMAGE_BASE + encodeURIComponent(productId) + ".jpg",
            productId,
            "product-image"
          );
          card.appendChild(productImage);

          const type = document.createElement("div");
          type.className = "product-type";
          type.textContent = productTypeLabel(productId);
          card.appendChild(type);

          const idEl = document.createElement("div");
          idEl.className = "asset-id";
          idEl.textContent = productId;
          card.appendChild(idEl);

          productsGrid.appendChild(card);
        }}

        section.appendChild(productsGrid);
        fragment.appendChild(section);
      }}

      summary.textContent = bundles.length + " bundles | " + totalProducts + " productos";
      gallery.appendChild(fragment);
      uploadScreen.style.display = "none";
      galleryRoot.style.display = "block";
      window.scrollTo({{ top: 0, behavior: "smooth" }});
    }}

    csvInput.addEventListener("change", async (event) => {{
      const file = event.target.files && event.target.files[0];
      if (!file) {{
        return;
      }}
      try {{
        const text = await file.text();
        const grouped = parseCsvRows(text);
        renderGrouped(grouped);
      }} catch (err) {{
        alert("Error leyendo CSV: " + (err && err.message ? err.message : String(err)));
      }}
    }});
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    product_descriptions = load_product_descriptions()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    html_content = build_html(product_descriptions)
    args.output.write_text(html_content, encoding="utf-8")

    print(f"Creator HTML written to: {args.output}")
    print(f"Product descriptions loaded: {len(product_descriptions)}")
    print(f"Descriptions source: {PRODUCTS_CSV}")


if __name__ == "__main__":
    main()
